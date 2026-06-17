"""
Color Validator Service

Evaluates:
  1. LAB Color Conversion — perceptually uniform color space
  2. Histogram Comparison — distribution-level similarity
  3. Delta E Analysis — per-pixel color difference

Color Score = 0.7 * deltaE_score + 0.3 * histogram_similarity
"""

import numpy as np
import cv2
from scipy.stats import wasserstein_distance


def validate_color(
    source_image_path: str,
    rendered_image_path: str,
) -> dict:
    """
    Run full color validation.

    Returns:
        dict with 'score' (0-100) and 'details' dict.
    """
    img_src = _load_and_prepare(source_image_path)
    img_rnd = _load_and_prepare(rendered_image_path)

    # Convert to LAB
    lab_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2LAB)
    lab_rnd = cv2.cvtColor(img_rnd, cv2.COLOR_BGR2LAB)

    # Delta E
    delta_e_result = _delta_e_analysis(lab_src, lab_rnd)

    # Histogram comparison
    hist_result = _histogram_comparison(lab_src, lab_rnd)

    # Combined score
    score = round(
        0.7 * delta_e_result["score"] + 0.3 * hist_result["score"],
        1,
    )
    score = max(0, min(100, score))

    return {
        "score": score,
        "details": {
            "delta_e": delta_e_result,
            "histogram": hist_result,
        },
    }


# ──────────────── Helpers ────────────────


def _load_and_prepare(image_path: str, target_size: tuple = (256, 256)) -> np.ndarray:
    """Load and resize an image, compositing alpha over white."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Composite alpha over white
    if len(img.shape) == 3 and img.shape[-1] == 4:
        alpha = img[:, :, 3:4] / 255.0
        rgb = img[:, :, :3]
        white = np.ones_like(rgb) * 255
        img = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)

    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def _get_foreground_mask(image_path: str, target_size: tuple = (256, 256)) -> np.ndarray:
    """Get a binary foreground mask from image (alpha or threshold)."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is not None and len(img.shape) == 3 and img.shape[-1] == 4:
        mask = (img[:, :, 3] > 128).astype(np.uint8)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = (gray < 240).astype(np.uint8)

    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return mask


# ──────────────── Delta E ────────────────


def _delta_e_analysis(lab_src: np.ndarray, lab_rnd: np.ndarray) -> dict:
    """
    Compute Delta E (CIE76) between source and rendered images.
    ΔE = sqrt((L1-L2)² + (a1-a2)² + (b1-b2)²)
    """
    lab_src_f = lab_src.astype(np.float64)
    lab_rnd_f = lab_rnd.astype(np.float64)

    # Per-pixel Delta E
    delta_e = np.sqrt(np.sum((lab_src_f - lab_rnd_f) ** 2, axis=2))

    mean_delta_e = float(np.mean(delta_e))
    median_delta_e = float(np.median(delta_e))
    max_delta_e = float(np.max(delta_e))

    # Scoring: ΔE < 2 is perceptually indistinguishable
    # Map mean_delta_e to a 0-100 score
    # ΔE=0 → 100, ΔE=10 → 50, ΔE≥30 → 0
    if mean_delta_e <= 2:
        score = 100 - mean_delta_e * 2.5
    elif mean_delta_e <= 10:
        score = 95 - (mean_delta_e - 2) * 5.625
    elif mean_delta_e <= 30:
        score = 50 - (mean_delta_e - 10) * 2.5
    else:
        score = max(0, 10 - (mean_delta_e - 30) * 0.33)

    return {
        "score": round(score, 1),
        "metrics": {
            "mean_delta_e": round(mean_delta_e, 2),
            "median_delta_e": round(median_delta_e, 2),
            "max_delta_e": round(max_delta_e, 2),
        },
    }


# ──────────────── Histogram Comparison ────────────────


def _histogram_comparison(lab_src: np.ndarray, lab_rnd: np.ndarray) -> dict:
    """Compare LAB color histograms using correlation and Earth Mover's Distance."""
    correlations = []
    emd_distances = []

    # Compare each LAB channel
    channel_names = ["L", "A", "B"]
    for i, name in enumerate(channel_names):
        hist_src = cv2.calcHist([lab_src], [i], None, [256], [0, 256]).flatten()
        hist_rnd = cv2.calcHist([lab_rnd], [i], None, [256], [0, 256]).flatten()

        # Normalize
        hist_src = hist_src / (hist_src.sum() + 1e-10)
        hist_rnd = hist_rnd / (hist_rnd.sum() + 1e-10)

        # Correlation (1 = perfect, -1 = inverse)
        corr = cv2.compareHist(
            hist_src.astype(np.float32),
            hist_rnd.astype(np.float32),
            cv2.HISTCMP_CORREL,
        )
        correlations.append(float(corr))

        # Earth Mover's Distance
        emd = wasserstein_distance(hist_src, hist_rnd)
        emd_distances.append(float(emd))

    avg_corr = np.mean(correlations)
    avg_emd = np.mean(emd_distances)

    # Scoring
    corr_score = max(0, avg_corr * 100)  # correlation ranges -1 to 1
    emd_score = max(0, 100 - avg_emd * 5000)  # scale EMD to 0-100

    score = 0.6 * corr_score + 0.4 * emd_score

    return {
        "score": round(score, 1),
        "metrics": {
            "avg_correlation": round(avg_corr, 4),
            "avg_emd": round(avg_emd, 6),
            "per_channel_correlation": {
                name: round(c, 4) for name, c in zip(channel_names, correlations)
            },
            "per_channel_emd": {
                name: round(e, 6) for name, e in zip(channel_names, emd_distances)
            },
        },
    }
