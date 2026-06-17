"""
Camera Pose Estimator (Enhancement 1)

Finds the camera viewpoint from which a rendered GLB best matches the
silhouette of the background-removed input image. All subsequent geometry,
texture, and color validation should use the resulting aligned render so the
image and the 3D model are compared from the same viewing angle.

Method:
  1. Extract a binary silhouette mask from the input image.
  2. Render the GLB across a grid of azimuth/elevation candidates.
  3. Score each candidate by silhouette IoU + contour overlap.
  4. Keep the highest-scoring pose and its aligned render.
"""

import os
import uuid
import numpy as np
import cv2

from services.glb_renderer import render_glb_from_pose, extract_silhouette_mask

# Default search space (degrees).
DEFAULT_AZIMUTHS = list(range(0, 360, 30))          # 0,30,...,330
DEFAULT_ELEVATIONS = [-15, 0, 15, 30]

MASK_SIZE = (256, 256)


def estimate_pose(
    image_path: str,
    glb_path: str,
    output_dir: str,
    azimuths=None,
    elevations=None,
    resolution: tuple = (512, 512),
) -> dict:
    """
    Search candidate viewpoints and return the best-aligned render.

    Returns dict:
        {
          "azimuth": int, "elevation": int, "iou": float,
          "contour_overlap": float, "confidence": float,
          "aligned_render_path": str, "aligned_render_url": str,
          "input_mask_path": str, "input_mask_url": str,
          "candidates_evaluated": int,
          "search_space": {...}
        }
    """
    azimuths = azimuths or DEFAULT_AZIMUTHS
    elevations = elevations or DEFAULT_ELEVATIONS

    # Step 1: input silhouette mask
    input_mask = extract_silhouette_mask(image_path, MASK_SIZE)
    input_mask_path = _save_mask(input_mask, output_dir, "input_mask")

    best = None  # (similarity, iou, contour, az, el, render_path)

    # Steps 2-4: render each candidate, score silhouette similarity
    for el in elevations:
        for az in azimuths:
            try:
                render_path = render_glb_from_pose(
                    glb_path, az, el, output_dir, resolution=resolution, suffix="cand"
                )
                render_mask = extract_silhouette_mask(render_path, MASK_SIZE)
            except Exception:
                continue

            iou = _iou(input_mask, render_mask)
            contour = _contour_overlap(input_mask, render_mask)
            similarity = 0.7 * iou + 0.3 * contour

            if best is None or similarity > best[0]:
                # Remove the previous best render to avoid littering outputs.
                if best is not None:
                    _safe_remove(best[5])
                best = (similarity, iou, contour, az, el, render_path)
            else:
                _safe_remove(render_path)

    if best is None:
        # Could not render any candidate — fall back to a single front render.
        aligned = render_glb_from_pose(glb_path, 0, 0, output_dir, resolution=resolution, suffix="aligned")
        return {
            "azimuth": 0,
            "elevation": 0,
            "iou": 0.0,
            "contour_overlap": 0.0,
            "confidence": 0.0,
            "aligned_render_path": aligned,
            "aligned_render_url": f"/outputs/{os.path.basename(aligned)}",
            "input_mask_path": input_mask_path,
            "input_mask_url": f"/outputs/{os.path.basename(input_mask_path)}",
            "candidates_evaluated": 0,
            "search_space": {"azimuths": azimuths, "elevations": elevations},
            "fallback": True,
        }

    similarity, iou, contour, az, el, render_path = best

    # Step 6: promote the winning render to a stable aligned_render name
    aligned_path = _rename_aligned(render_path, output_dir)

    return {
        "azimuth": az,
        "elevation": el,
        "iou": round(float(iou), 4),
        "contour_overlap": round(float(contour), 4),
        "confidence": round(float(similarity), 4),
        "aligned_render_path": aligned_path,
        "aligned_render_url": f"/outputs/{os.path.basename(aligned_path)}",
        "input_mask_path": input_mask_path,
        "input_mask_url": f"/outputs/{os.path.basename(input_mask_path)}",
        "candidates_evaluated": len(azimuths) * len(elevations),
        "search_space": {"azimuths": azimuths, "elevations": elevations},
        "fallback": False,
    }


# ──────────────── Similarity metrics ────────────────


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def _contour_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Symmetric contour-band overlap: dilate each contour into a thin band and
    measure how much of one band falls on the other. Rewards edge alignment
    even when filled IoU is similar.
    """
    edges_a = _contour_band(mask_a)
    edges_b = _contour_band(mask_b)
    if edges_a.sum() == 0 or edges_b.sum() == 0:
        return 0.0
    a_on_b = np.logical_and(edges_a, edges_b).sum() / max(edges_a.sum(), 1)
    b_on_a = np.logical_and(edges_b, edges_a).sum() / max(edges_b.sum(), 1)
    return float((a_on_b + b_on_a) / 2)


def _contour_band(mask: np.ndarray, thickness: int = 3) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    band = np.zeros_like(mask)
    cv2.drawContours(band, contours, -1, 255, thickness)
    return band > 0


# ──────────────── File helpers ────────────────


def _save_mask(mask: np.ndarray, output_dir: str, suffix: str) -> str:
    path = os.path.join(output_dir, f"{uuid.uuid4()}_{suffix}.png")
    cv2.imwrite(path, mask)
    return path


def _rename_aligned(render_path: str, output_dir: str) -> str:
    aligned_path = os.path.join(output_dir, f"{uuid.uuid4()}_aligned_render.png")
    try:
        os.replace(render_path, aligned_path)
        return aligned_path
    except Exception:
        return render_path


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
