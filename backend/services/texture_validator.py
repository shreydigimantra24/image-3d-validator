"""
Texture Validator Service

Evaluates:
  1. Texture Presence (materials, UV coords, texture files)
  2. Perceptual Comparison (SSIM, LPIPS)

Texture Score = 0.2 * texture_presence + 0.4 * ssim_score + 0.4 * lpips_score

Perceptual metrics are computed on the SAME non-dark appearance image the color
module uses (color_validator.prepare_comparison) — the render with luminance
normalized to the source, NEVER the raw dark metallic render (Fix B). This keeps
the texture and color panels from contradicting each other.
"""

import numpy as np
import trimesh
import cv2
from skimage.metrics import structural_similarity as ssim

from services.mesh_cache import load_scene
from services.validation_config import IOU_TRUST_THRESHOLD
from services.color_validator import prepare_comparison

# Cached LPIPS network. Instantiating lpips.LPIPS(net="alex") per call reloads
# AlexNet weights and leaks CPU/VRAM (PyTorch's caching allocator does not
# return GPU memory to the OS), exhausting VRAM in a few requests.
_lpips_model = None


def _get_lpips_model():
    """Lazily build and cache the LPIPS network once per process."""
    global _lpips_model
    if _lpips_model is None:
        import lpips

        _lpips_model = lpips.LPIPS(net="alex")
    return _lpips_model


def validate_texture(
    source_image_path: str,
    rendered_image_path: str,
    glb_path: str,
    alignment: dict = None,
    albedo_rgb: "np.ndarray|None" = None,
) -> dict:
    """
    Run full texture validation.

    Per-pixel SSIM/LPIPS only compare meaningfully when the render and photo
    silhouettes overlap tightly (Fix 4). When alignment IoU is below
    IOU_TRUST_THRESHOLD we DO NOT trust SSIM/LPIPS; instead we score on an
    alignment-robust foreground LAB histogram and report reduced confidence.

    All perceptual metrics run on the shared non-dark appearance image
    (color_validator.prepare_comparison) — the exposure-normalized render, never
    the raw dark metallic render (Fix B).

    Args:
        alignment: optional pose result carrying 'iou' / 'confidence'. If None,
            alignment is assumed adequate (legacy behaviour).
        albedo_rgb: optional albedo texture, forwarded to prepare_comparison so
            texture and color share the same comparison basis.

    Returns dict with 'score' (0-100), 'confidence' (0-1) and 'details'.
    """
    presence = _texture_presence(glb_path)

    # SAME non-dark basis as the color module.
    basis = prepare_comparison(source_image_path, rendered_image_path, albedo_rgb)
    src_bgr, src_mask = basis["src_bgr"], basis["src_mask"]
    app_bgr, app_mask = basis["appearance_bgr"], basis["appearance_mask"]

    # FOREGROUND mask for per-pixel metrics. SSIM/LPIPS over the FULL frame are
    # dominated by the (matching white) background, which inflates the score for
    # ANY pair — even a wrong model — and destroys discrimination. We therefore
    # compute every per-pixel metric on the foreground only (Fix 6): the UNION of
    # the two foreground masks, so the overlapping region is compared as real
    # surface texture while non-overlapping pixels (one foreground, one bg) count
    # as genuine differences and penalise a shape/object mismatch.
    fg_union = ((src_mask > 0) | (app_mask > 0)).astype(np.uint8)

    iou = None if alignment is None else alignment.get("iou")
    trusted = iou is None or float(iou) >= IOU_TRUST_THRESHOLD

    perceptual = _perceptual_comparison(src_bgr, app_bgr, fg_union, src_mask, app_mask)

    if trusted:
        score = 0.2 * presence["score"] + 0.8 * perceptual["appearance_score"]
        metric_used = "foreground_masked_ssim+lpips+hist"
        confidence = 1.0 if iou is None else round(float(iou), 3)
    else:
        # Very low IoU → even the masked metrics compare too little overlapping
        # content to trust. Fall back to the translation-robust foreground
        # LAB-histogram similarity (position-invariant). Presence weighted LOW
        # (0.2): "the GLB has materials/UVs/texture" is true of ANY valid asset.
        hist_sim = _foreground_hist_similarity(src_bgr, src_mask, app_bgr, app_mask)
        score = 0.2 * presence["score"] + 0.8 * hist_sim
        metric_used = "foreground_lab_histogram (low-IoU fallback)"
        confidence = round(float(iou) * 0.6, 3)
        perceptual["histogram_similarity"] = round(hist_sim, 1)

    perceptual["metric_used"] = metric_used
    perceptual["trusted"] = trusted
    perceptual["alignment_iou"] = None if iou is None else round(float(iou), 4)
    perceptual["comparison_basis"] = basis["appearance_basis"]

    score = round(max(0, min(100, score)), 1)
    return {
        "score": score,
        "confidence": confidence,
        "details": {
            "texture_presence": presence,
            "perceptual": perceptual,
        },
    }


# ──────────────── Texture Presence ────────────────


def _texture_presence(glb_path: str) -> dict:
    """Check if the GLB has proper materials, textures, and UV mapping."""
    scene = load_scene(glb_path)

    has_material = False
    has_texture = False
    has_uv = False
    material_count = 0
    texture_count = 0

    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue

        # Check material
        if geom.visual is not None:
            material = getattr(geom.visual, "material", None)
            if material is not None:
                has_material = True
                material_count += 1

                # Check texture image
                if hasattr(material, "image") and material.image is not None:
                    has_texture = True
                    texture_count += 1
                elif hasattr(material, "baseColorTexture") and material.baseColorTexture is not None:
                    has_texture = True
                    texture_count += 1

            # Check UV coordinates
            if hasattr(geom.visual, "uv") and geom.visual.uv is not None:
                if len(geom.visual.uv) > 0:
                    has_uv = True

    # Score components
    material_score = 100 if has_material else 0
    texture_score = 100 if has_texture else 0
    uv_score = 100 if has_uv else 0

    presence_score = 0.3 * material_score + 0.4 * texture_score + 0.3 * uv_score

    return {
        "score": round(presence_score, 1),
        "checks": {
            "has_material": has_material,
            "has_texture": has_texture,
            "has_uv_coordinates": has_uv,
            "material_count": material_count,
            "texture_count": texture_count,
        },
    }


# ──────────────── Perceptual Comparison ────────────────
#
# All helpers operate on prepared BGR image ARRAYS (the shared non-dark
# appearance basis) rather than re-reading the raw render from disk (Fix B).


def _compute_ssim(src_bgr: np.ndarray, model_bgr: np.ndarray,
                  mask: "np.ndarray|None" = None) -> float:
    """Structural Similarity between two prepared BGR images.

    When `mask` is given, the SSIM map is averaged over the masked (foreground)
    pixels ONLY — the matching white background is excluded so it can't inflate
    the score (Fix 6). Pixels where the masks disagree are still compared, so a
    shape mismatch correctly lowers SSIM there.
    """
    gray1 = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray2 = cv2.cvtColor(model_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    score, smap = ssim(gray1, gray2, full=True, data_range=255)
    if mask is None:
        return float(score)
    m = mask > 0
    if m.sum() == 0:
        return float(score)
    return float(np.clip(smap[m].mean(), -1.0, 1.0))


def _compute_lpips(src_bgr: np.ndarray, model_bgr: np.ndarray,
                   mask: "np.ndarray|None" = None) -> float:
    """
    LPIPS perceptual distance between two prepared BGR images.

    When `mask` is given, the background of BOTH images is flattened to a common
    neutral gray so LPIPS focuses on the FOREGROUND surface, not the (identical
    white) background that would otherwise drag the distance toward 0 for any
    pair (Fix 6). Falls back to a masked pixel-difference proxy if LPIPS missing.
    """
    a = src_bgr
    b = model_bgr
    if mask is not None:
        m = mask > 0
        a = src_bgr.copy()
        b = model_bgr.copy()
        a[~m] = 127
        b[~m] = 127
    try:
        import torch

        loss_fn = _get_lpips_model()

        # Convert BGR to RGB and normalize to [-1, 1]
        t1 = torch.from_numpy(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 127.5 - 1.0
        t2 = torch.from_numpy(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 127.5 - 1.0

        t1 = t1.unsqueeze(0)
        t2 = t2.unsqueeze(0)

        with torch.no_grad():
            distance = loss_fn(t1, t2).item()

        return distance

    except ImportError:
        # Fallback: masked normalized pixel-level difference as proxy.
        af = a.astype(np.float32) / 255.0
        bf = b.astype(np.float32) / 255.0
        if mask is not None and (mask > 0).sum() > 0:
            mm = mask > 0
            mse = np.mean(((af - bf) ** 2)[mm])
        else:
            mse = np.mean((af - bf) ** 2)
        return float(np.clip(mse * 2, 0, 1))  # Scale to roughly match LPIPS range


def _foreground_texture_similarity(src_bgr, src_mask, ref_bgr, ref_mask) -> float:
    """
    Alignment-robust 0-100 TEXTURE/pattern similarity.

    Compares magnitude-weighted gradient-orientation histograms of the two
    foregrounds (a HOG-style descriptor) plus the gradient-magnitude
    distribution. This captures "is this the same kind of surface detail / weave /
    print" WITHOUT needing pixel correspondence, so — unlike SSIM — a few pixels
    of render-vs-photo misalignment does not tank a genuine match. Pairs with the
    color histogram (which carries hue/chroma) to form a precise yet
    pose-tolerant surface-match signal.
    """
    def descriptor(bgr, mask):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        ang = np.mod(np.arctan2(gy, gx), np.pi)  # 0..pi orientation
        m = mask > 0
        if m.sum() == 0:
            m = np.ones(gray.shape, bool)
        oh, _ = np.histogram(ang[m], bins=18, range=(0, np.pi), weights=mag[m])
        oh = oh / (oh.sum() + 1e-10)
        # magnitude distribution (detail "busy-ness"), normalized.
        mh, _ = np.histogram(mag[m], bins=16, range=(0, 255))
        mh = mh / (mh.sum() + 1e-10)
        return oh.astype(np.float32), mh.astype(np.float32)

    o_s, m_s = descriptor(src_bgr, src_mask)
    o_r, m_r = descriptor(ref_bgr, ref_mask)
    o_corr = cv2.compareHist(o_s, o_r, cv2.HISTCMP_CORREL)
    m_corr = cv2.compareHist(m_s, m_r, cv2.HISTCMP_CORREL)
    return float(np.clip((0.5 * o_corr + 0.5 * m_corr) * 100.0, 0, 100))


# LAB channel weights for the color cue. The L (lightness) channel mostly
# reflects LIGHTING/shading — a lit render spreads its lightness differently than
# a photo even when the material color is identical — so it is heavily
# down-weighted. The chroma channels (A=green-red, B=blue-yellow) carry the true
# material color and discriminate a real color mismatch (e.g. blue vs grey), so
# they dominate. This keeps a genuine match's color cue HIGH while a wrong color
# still scores LOW.
_LAB_CHANNEL_WEIGHTS = (0.2, 0.4, 0.4)  # (L, A, B)


def _foreground_hist_similarity(src_bgr, src_mask, model_bgr, model_mask) -> float:
    """
    Alignment-robust 0-100 color similarity: weighted correlation of foreground
    LAB histograms (chroma-dominant — see _LAB_CHANNEL_WEIGHTS). Does not require
    pixel correspondence, so it stays meaningful when the pose IoU is too low to
    trust SSIM/LPIPS, and it is not dragged down by lighting/shading differences.
    """
    def fg_lab_hist(bgr, mask):
        m = (mask > 0).astype(np.uint8)
        if m.sum() == 0:
            m = np.ones(bgr.shape[:2], np.uint8)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        hists = []
        for ch in range(3):
            h = cv2.calcHist([lab], [ch], m, [64], [0, 256]).flatten()
            h = h / (h.sum() + 1e-10)
            hists.append(h.astype(np.float32))
        return hists

    hs = fg_lab_hist(src_bgr, src_mask)
    hr = fg_lab_hist(model_bgr, model_mask)
    corrs = [cv2.compareHist(a, b, cv2.HISTCMP_CORREL) for a, b in zip(hs, hr)]
    weighted = sum(w * c for w, c in zip(_LAB_CHANNEL_WEIGHTS, corrs))
    return float(np.clip(weighted * 100.0, 0, 100))


def _perceptual_comparison(src_bgr: np.ndarray, model_bgr: np.ndarray,
                           fg_mask: "np.ndarray|None" = None,
                           src_mask: "np.ndarray|None" = None,
                           app_mask: "np.ndarray|None" = None) -> dict:
    """Foreground-masked surface-appearance comparison (Fix 6).

    Blends FOUR foreground-only cues into a single `appearance_score` (0-100),
    leading with the two that are robust to render-vs-photo misalignment so a
    genuine surface match scores high even when the pixels don't register
    perfectly:
      * foreground LAB-histogram correlation (0.40) — color/material distribution,
        position-invariant. Carries hue/chroma (gray-vs-blue etc.).
      * foreground gradient/texture descriptor (0.35) — pattern / weave / detail
        statistics, position-invariant. Carries surface structure.
      * masked LPIPS (0.15) — deep perceptual similarity over the foreground.
      * masked SSIM  (0.10) — local pixel structure; a BONUS when alignment is
        tight, deliberately down-weighted because SSIM collapses under a few
        pixels of misalignment and would otherwise cap a correct match.

    All cues exclude the background, so the score reflects how much the model's
    SURFACE matches the photographed surface, not how much white margin the two
    images share.
    """
    ssim_val = _compute_ssim(src_bgr, model_bgr, fg_mask)
    lpips_val = _compute_lpips(src_bgr, model_bgr, fg_mask)

    # Convert to 0-100 scores
    ssim_score = max(0.0, ssim_val * 100)          # SSIM is -1..1, higher better
    lpips_score = max(0.0, (1 - lpips_val) * 100)  # LPIPS is 0..1, lower better

    # Position-invariant cues (use each image's own foreground mask).
    sm = src_mask if src_mask is not None else (fg_mask if fg_mask is not None else np.ones(src_bgr.shape[:2], np.uint8))
    am = app_mask if app_mask is not None else (fg_mask if fg_mask is not None else np.ones(model_bgr.shape[:2], np.uint8))
    hist_sim = _foreground_hist_similarity(src_bgr, sm, model_bgr, am)
    tex_sim = _foreground_texture_similarity(src_bgr, sm, model_bgr, am)

    appearance_score = round(
        0.40 * hist_sim + 0.35 * tex_sim + 0.15 * lpips_score + 0.10 * ssim_score, 1
    )

    return {
        "ssim_raw": round(ssim_val, 4),
        "lpips_raw": round(lpips_val, 4),
        "ssim_score": round(ssim_score, 1),
        "lpips_score": round(lpips_score, 1),
        "histogram_similarity": round(hist_sim, 1),
        "texture_pattern_similarity": round(tex_sim, 1),
        "appearance_score": appearance_score,
    }
