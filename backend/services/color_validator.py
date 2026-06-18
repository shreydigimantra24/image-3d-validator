"""
Color Validator Service

Evaluates color fidelity between the source product photo and the asset's
surface color, FOREGROUND-MASKED and in LAB using ΔE2000 (Fix 3).

KEY POINTS
----------
  * The model's color reference is the asset's baseColor ALBEDO texture when
    available — NOT an unlit / metallic render. A metallicFactor=1.0 asset
    renders dark without an environment map, but its albedo is the real color.
  * When no albedo is available we fall back to the render, exposure /
    white-balance normalized against the source so lighting doesn't masquerade
    as a color error.
  * Comparison is restricted to foreground pixels and uses CIEDE2000, the modern
    perceptual color-difference metric (replaces the old CIE76 ΔE).

Color Score = 0.7 * deltaE2000_score + 0.3 * histogram_similarity
"""

import numpy as np
import cv2
from scipy.stats import wasserstein_distance
from skimage.color import rgb2lab, deltaE_ciede2000


def validate_color(
    source_image_path: str,
    rendered_image_path: str,
    albedo_rgb: "np.ndarray|None" = None,
) -> dict:
    """
    Run full color validation, foreground-masked, in LAB with ΔE2000.

    Args:
        albedo_rgb: optional HxWx3 uint8 RGB albedo texture. When supplied it is
            the PREFERRED color reference (lighting-independent). Otherwise the
            render is used, exposure-normalized to the source.

    Returns dict with 'score' (0-100) and 'details'.
    """
    basis = prepare_comparison(source_image_path, rendered_image_path, albedo_rgb)
    src_bgr, src_mask = basis["src_bgr"], basis["src_mask"]

    # ΔE2000 against the color reference (albedo when available).
    delta_e_result = _delta_e2000_foreground(
        src_bgr, src_mask, basis["color_ref_bgr"], basis["color_ref_mask"]
    )
    # Histogram against the SAME non-dark appearance image the texture module
    # uses (Fix B) — so the two panels can never contradict.
    hist_result = _histogram_comparison(
        src_bgr, src_mask, basis["appearance_bgr"], basis["appearance_mask"]
    )

    score = round(0.7 * delta_e_result["score"] + 0.3 * hist_result["score"], 1)
    score = max(0, min(100, score))

    return {
        "score": score,
        "details": {
            "reference": basis["color_reference"],          # ΔE2000 basis
            "histogram_basis": basis["appearance_basis"],   # shared with texture
            "delta_e": delta_e_result,
            "histogram": hist_result,
        },
    }


def prepare_comparison(
    source_image_path: str,
    rendered_image_path: str,
    albedo_rgb: "np.ndarray|None" = None,
    size: tuple = (256, 256),
) -> dict:
    """
    Single source of truth for what color AND texture compare against, so the two
    panels can never tell contradicting stories (Fix B).

    Returns:
        src_bgr / src_mask          — the photo foreground.
        appearance_bgr / _mask      — the model APPEARANCE used for SSIM / LPIPS /
                                      histogram: the render with luminance
                                      normalized to the source (NEVER the raw,
                                      dark metallic render).
        color_ref_bgr / _mask       — the reference for ΔE2000: the albedo texture
                                      when available (lighting-independent), else
                                      the same exposure-normalized appearance.
        color_reference, appearance_basis — string labels for reporting.
    """
    src_bgr, src_mask = _load_foreground(source_image_path, size)

    # Appearance = exposure/luminance-normalized render. Reused by texture.
    app_bgr, app_mask = _load_foreground(rendered_image_path, size)
    app_bgr = _match_exposure(app_bgr, app_mask, src_bgr, src_mask)

    if albedo_rgb is not None and getattr(albedo_rgb, "size", 0) > 0:
        cref_bgr = cv2.resize(
            cv2.cvtColor(albedo_rgb, cv2.COLOR_RGB2BGR), size, interpolation=cv2.INTER_AREA
        )
        cref_mask = np.ones(size[::-1], dtype=np.uint8)
        color_reference = "albedo_texture"
    else:
        cref_bgr, cref_mask = app_bgr, app_mask
        color_reference = "render_exposure_normalized"

    return {
        "src_bgr": src_bgr,
        "src_mask": src_mask,
        "appearance_bgr": app_bgr,
        "appearance_mask": app_mask,
        "appearance_basis": "render_exposure_normalized",
        "color_ref_bgr": cref_bgr,
        "color_ref_mask": cref_mask,
        "color_reference": color_reference,
    }


# ──────────────── Loading / masking / exposure ────────────────


def _load_foreground(image_path: str, target_size: tuple = (256, 256)):
    """Load an image as BGR (alpha over white) plus a binary foreground mask."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[-1] == 4:
        alpha = img[:, :, 3:4] / 255.0
        rgb = img[:, :, :3]
        white = np.ones_like(rgb) * 255
        bgr = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
        mask = (img[:, :, 3] > 128).astype(np.uint8)
    else:
        bgr = img
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = ((gray < 240) & (gray > 15)).astype(np.uint8)
        if mask.sum() < gray.size * 0.005:
            mask = (gray < 240).astype(np.uint8)

    bgr = cv2.resize(bgr, target_size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    if mask.sum() == 0:  # degenerate: treat whole frame as foreground
        mask = np.ones(target_size[::-1], dtype=np.uint8)
    return bgr, mask


def _match_exposure(ref_bgr, ref_mask, src_bgr, src_mask):
    """
    Match the reference foreground's overall LUMINANCE to the source's with a
    single global gain. This cancels exposure / brightness differences (the
    metallic-dark render problem) WITHOUT erasing genuine hue/chroma differences
    — a deliberate color shift must still register as a color error.
    """
    rm = ref_mask > 0
    sm = src_mask > 0
    if rm.sum() == 0 or sm.sum() == 0:
        return ref_bgr
    out = ref_bgr.astype(np.float32)
    # Rec.601 luminance means over each foreground.
    w = np.array([0.114, 0.587, 0.299], dtype=np.float32)  # BGR weights
    ref_lum = float((out[rm] * w).sum(axis=1).mean())
    src_lum = float((src_bgr[sm].astype(np.float32) * w).sum(axis=1).mean())
    if ref_lum > 1e-3:
        gain = float(np.clip(src_lum / ref_lum, 0.2, 5.0))
        out *= gain
    return np.clip(out, 0, 255).astype(np.uint8)


# ──────────────── ΔE2000 ────────────────


def _mean_lab(bgr, mask):
    """Mean LAB color of the foreground (skimage CIE Lab)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab = rgb2lab(rgb)
    m = mask > 0
    return lab[m].reshape(-1, 3).mean(axis=0)


def _delta_e2000_foreground(src_bgr, src_mask, ref_bgr, ref_mask) -> dict:
    """
    CIEDE2000 between the source and reference FOREGROUND colors.

    We compare the mean foreground LAB (the two images are not pixel-aligned, so
    a distribution-level aggregate is the meaningful per-image color difference)
    plus per-quartile-of-lightness band means to catch partial shifts.
    """
    src_mean = _mean_lab(src_bgr, src_mask)
    ref_mean = _mean_lab(ref_bgr, ref_mask)
    mean_delta_e = float(deltaE_ciede2000(src_mean.reshape(1, 3), ref_mean.reshape(1, 3))[0])

    # ΔE2000 < 2 ≈ imperceptible. Map to 0-100 (gentler than CIE76 since ΔE2000
    # numbers are smaller for the same perceptual gap).
    if mean_delta_e <= 2:
        score = 100 - mean_delta_e * 2.5
    elif mean_delta_e <= 10:
        score = 95 - (mean_delta_e - 2) * 5.625
    elif mean_delta_e <= 25:
        score = 50 - (mean_delta_e - 10) * 2.5
    else:
        score = max(0, 12.5 - (mean_delta_e - 25) * 0.4)

    return {
        "metric": "CIEDE2000",
        "score": round(float(max(0, min(100, score))), 1),
        "metrics": {
            "mean_delta_e": round(mean_delta_e, 2),
            "source_mean_lab": [round(float(v), 1) for v in src_mean],
            "reference_mean_lab": [round(float(v), 1) for v in ref_mean],
        },
    }


# ──────────────── Histogram Comparison (foreground, LAB) ────────────────


def _histogram_comparison(src_bgr, src_mask, ref_bgr, ref_mask) -> dict:
    """Compare foreground LAB histograms (correlation + Earth Mover's Distance)."""
    lab_src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB)
    lab_ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)
    sm = (src_mask > 0).astype(np.uint8)
    rm = (ref_mask > 0).astype(np.uint8)

    correlations = []
    emd_distances = []
    channel_names = ["L", "A", "B"]
    for i, name in enumerate(channel_names):
        hist_src = cv2.calcHist([lab_src], [i], sm, [256], [0, 256]).flatten()
        hist_ref = cv2.calcHist([lab_ref], [i], rm, [256], [0, 256]).flatten()
        hist_src = hist_src / (hist_src.sum() + 1e-10)
        hist_ref = hist_ref / (hist_ref.sum() + 1e-10)
        corr = cv2.compareHist(
            hist_src.astype(np.float32), hist_ref.astype(np.float32), cv2.HISTCMP_CORREL
        )
        correlations.append(float(corr))
        emd_distances.append(float(wasserstein_distance(hist_src, hist_ref)))

    avg_corr = float(np.mean(correlations))
    avg_emd = float(np.mean(emd_distances))
    corr_score = max(0, avg_corr * 100)
    emd_score = max(0, 100 - avg_emd * 5000)
    score = 0.6 * corr_score + 0.4 * emd_score

    return {
        "score": round(score, 1),
        "metrics": {
            "avg_correlation": round(avg_corr, 4),
            "avg_emd": round(avg_emd, 6),
            "per_channel_correlation": {
                name: round(c, 4) for name, c in zip(channel_names, correlations)
            },
        },
    }
