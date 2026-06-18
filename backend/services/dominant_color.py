"""
Dominant Color Analysis (Enhancement 5)

Histogram comparison can miss perceptually important shifts (e.g. a product's
primary color drifting red → orange). This module isolates foreground pixels,
clusters them with KMeans, and measures the perceptual distance between the
input image's dominant palette and the aligned render's palette in LAB space.
"""

import numpy as np
import cv2
from skimage.color import deltaE_ciede2000

N_CLUSTERS = 5

# KMeans on a downscaled image yields a near-identical palette. Resizing up
# front avoids allocating a multi-million-pixel foreground array from 4K inputs.
PALETTE_MAX_SIZE = 256


def analyze_dominant_colors(
    source_image_path: str,
    rendered_image_path: str,
    n_clusters: int = N_CLUSTERS,
    render_rgb: "np.ndarray|None" = None,
) -> dict:
    """
    Cluster foreground colors of both images and match dominant palettes in LAB
    using ΔE2000.

    Args:
        render_rgb: optional HxWx3 uint8 RGB array used as the model's color
            reference INSTEAD of loading rendered_image_path — pass the asset's
            albedo texture here so the palette match is lighting-independent.

    Returns dict:
        {
          "dominant_color_distance": float,   # weighted mean ΔE2000 of matched palette
          "source_palette": [{"rgb":[r,g,b], "weight":float}, ...],
          "render_palette": [...],
          "primary_shift": {"source_rgb":[...], "render_rgb":[...], "delta_e":float}
        }
    """
    src_palette = _dominant_palette(source_image_path, n_clusters)
    if render_rgb is not None and render_rgb.size > 0:
        rnd_palette = _dominant_palette_from_rgb(render_rgb, n_clusters)
    else:
        rnd_palette = _dominant_palette(rendered_image_path, n_clusters)

    if not src_palette or not rnd_palette:
        return {
            "dominant_color_distance": 100.0,
            "source_palette": _palette_to_json(src_palette),
            "render_palette": _palette_to_json(rnd_palette),
            "primary_shift": None,
        }

    # Weighted greedy match: pair each source cluster with nearest render cluster.
    distance = _matched_palette_distance(src_palette, rnd_palette)

    # Primary color shift = the most weighted cluster on each side.
    src_primary = max(src_palette, key=lambda c: c["weight"])
    rnd_primary = max(rnd_palette, key=lambda c: c["weight"])
    primary_de = _delta_e_lab(src_primary["rgb"], rnd_primary["rgb"])

    return {
        "dominant_color_distance": round(float(distance), 2),
        "source_palette": _palette_to_json(src_palette),
        "render_palette": _palette_to_json(rnd_palette),
        "primary_shift": {
            "source_rgb": [int(v) for v in src_primary["rgb"]],
            "render_rgb": [int(v) for v in rnd_primary["rgb"]],
            "delta_e": round(float(primary_de), 2),
        },
    }


def dominant_color_score(distance: float) -> float:
    """Map dominant-color LAB ΔE to a 0-100 score (ΔE 0→100, 50+→0)."""
    return round(max(0.0, 100.0 - float(distance) * 2.0), 1)


# ──────────────── Clustering ────────────────


def _dominant_palette(image_path: str, n_clusters: int):
    """Return [{"rgb": np.array, "weight": float}] for foreground pixels."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return []

    # Downscale BEFORE masking/pixel extraction so the huge foreground array is
    # never allocated for high-resolution inputs.
    h, w = img.shape[:2]
    scale = PALETTE_MAX_SIZE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(
            img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )

    mask = _foreground_mask(img)

    # Composite to BGR.
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[:, :, :3]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pixels = rgb[mask > 0].astype(np.float32)
    return _cluster_pixels(pixels, n_clusters)


def _dominant_palette_from_rgb(rgb: np.ndarray, n_clusters: int):
    """Palette of an in-memory RGB array (e.g. the albedo texture). Treats every
    pixel as foreground — albedo is pure surface color."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return []
    pixels = rgb[:, :, :3].reshape(-1, 3).astype(np.float32)
    return _cluster_pixels(pixels, n_clusters)


def _cluster_pixels(pixels: np.ndarray, n_clusters: int):
    """KMeans a set of RGB pixels into a weighted palette."""
    if len(pixels) < n_clusters:
        return []

    # Subsample for speed on large images.
    if len(pixels) > 20000:
        idx = np.linspace(0, len(pixels) - 1, 20000).astype(int)
        pixels = pixels[idx]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()
    counts = np.bincount(labels, minlength=n_clusters).astype(np.float32)
    weights = counts / counts.sum()

    palette = []
    for i in range(n_clusters):
        palette.append({"rgb": centers[i], "weight": float(weights[i])})
    return palette


def _foreground_mask(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[-1] == 4:
        return (img[:, :, 3] > 128).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = ((gray < 240) & (gray > 15)).astype(np.uint8)
    if mask.sum() < gray.size * 0.005:
        mask = (gray < 240).astype(np.uint8)
    return mask


# ──────────────── Distance ────────────────


def _matched_palette_distance(src_palette, rnd_palette) -> float:
    """Weighted mean ΔE of each source cluster to its nearest render cluster."""
    total_w = 0.0
    total_d = 0.0
    for c in src_palette:
        nearest = min(rnd_palette, key=lambda r: _delta_e_lab(c["rgb"], r["rgb"]))
        d = _delta_e_lab(c["rgb"], nearest["rgb"])
        total_d += d * c["weight"]
        total_w += c["weight"]
    return total_d / max(total_w, 1e-9)


def _delta_e_lab(rgb_a, rgb_b) -> float:
    """CIEDE2000 ΔE between two RGB colors (perceptually accurate; replaces CIE76)."""
    lab_a = _rgb_to_lab(rgb_a)
    lab_b = _rgb_to_lab(rgb_b)
    return float(deltaE_ciede2000(lab_a.reshape(1, 3), lab_b.reshape(1, 3))[0])


def _rgb_to_lab(rgb) -> np.ndarray:
    """Standard CIE Lab (L 0-100) via skimage — matches deltaE_ciede2000's domain.
    (cv2's LAB is 0-255 scaled and would corrupt ΔE2000.)"""
    from skimage.color import rgb2lab

    arr = np.clip(np.array(rgb, dtype=np.float32).reshape(1, 1, 3), 0, 255) / 255.0
    return rgb2lab(arr).astype(np.float64).reshape(3)


def _palette_to_json(palette):
    return [
        {"rgb": [int(v) for v in c["rgb"]], "weight": round(float(c["weight"]), 3)}
        for c in palette
    ]
