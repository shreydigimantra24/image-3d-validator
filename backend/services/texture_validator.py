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

    iou = None if alignment is None else alignment.get("iou")
    trusted = iou is None or float(iou) >= IOU_TRUST_THRESHOLD

    perceptual = _perceptual_comparison(src_bgr, app_bgr)

    if trusted:
        score = (
            0.2 * presence["score"]
            + 0.4 * perceptual["ssim_score"]
            + 0.4 * perceptual["lpips_score"]
        )
        metric_used = "ssim+lpips"
        confidence = 1.0 if iou is None else round(float(iou), 3)
    else:
        # Low IoU → SSIM/LPIPS compare partly non-overlapping content. Fall back
        # to a translation-robust foreground LAB-histogram similarity.
        hist_sim = _foreground_hist_similarity(src_bgr, src_mask, app_bgr, app_mask)
        score = 0.5 * presence["score"] + 0.5 * hist_sim
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


def _compute_ssim(src_bgr: np.ndarray, model_bgr: np.ndarray) -> float:
    """Compute Structural Similarity Index between two prepared BGR images."""
    gray1 = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(model_bgr, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return float(score)


def _compute_lpips(src_bgr: np.ndarray, model_bgr: np.ndarray) -> float:
    """
    Compute LPIPS perceptual distance between two prepared BGR images.
    Falls back to a simpler perceptual metric if LPIPS is unavailable.
    """
    try:
        import torch

        loss_fn = _get_lpips_model()

        # Convert BGR to RGB and normalize to [-1, 1]
        t1 = torch.from_numpy(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 127.5 - 1.0
        t2 = torch.from_numpy(cv2.cvtColor(model_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 127.5 - 1.0

        t1 = t1.unsqueeze(0)
        t2 = t2.unsqueeze(0)

        with torch.no_grad():
            distance = loss_fn(t1, t2).item()

        return distance

    except ImportError:
        # Fallback: use normalized pixel-level difference as proxy
        a = src_bgr.astype(np.float32) / 255.0
        b = model_bgr.astype(np.float32) / 255.0
        mse = np.mean((a - b) ** 2)
        return float(np.clip(mse * 2, 0, 1))  # Scale to roughly match LPIPS range


def _foreground_hist_similarity(src_bgr, src_mask, model_bgr, model_mask) -> float:
    """
    Alignment-robust 0-100 similarity: correlation of foreground LAB histograms
    of two prepared BGR images + masks. Does not require pixel correspondence, so
    it stays meaningful when the pose IoU is too low to trust SSIM/LPIPS.
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
    return float(np.clip(np.mean(corrs) * 100.0, 0, 100))


def _perceptual_comparison(src_bgr: np.ndarray, model_bgr: np.ndarray) -> dict:
    """Run SSIM and LPIPS comparisons on prepared BGR image arrays."""
    ssim_val = _compute_ssim(src_bgr, model_bgr)
    lpips_val = _compute_lpips(src_bgr, model_bgr)

    # Convert to 0-100 scores
    ssim_score = ssim_val * 100  # SSIM is 0-1, higher is better
    lpips_score = max(0, (1 - lpips_val) * 100)  # LPIPS is 0-1, lower is better

    return {
        "ssim_raw": round(ssim_val, 4),
        "lpips_raw": round(lpips_val, 4),
        "ssim_score": round(ssim_score, 1),
        "lpips_score": round(lpips_score, 1),
    }
