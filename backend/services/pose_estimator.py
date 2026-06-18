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
import gc
import logging
import numpy as np
import cv2

from services.glb_renderer import render_glb_from_pose, extract_silhouette_mask, PoseRenderer

logger = logging.getLogger(__name__)

# Coarse search space (degrees). 45° azimuth step keeps the candidate count
# (and memory/CPU) bounded while still covering all sides of the object.
DEFAULT_AZIMUTHS = list(range(0, 360, 45))          # 0,45,...,315  (8)
DEFAULT_ELEVATIONS = [-15, 0, 15, 30]               # (4)  → 32 candidates

# Fine search: a dense grid centred on the coarse winner. Offsets in degrees.
# 9 azimuth × 5 elevation = 45 extra candidates → ~77 total, one GLB load.
FINE_AZIMUTH_OFFSETS = list(range(-20, 21, 5))      # -20..+20 step 5 (9)
FINE_ELEVATION_OFFSETS = list(range(-10, 11, 5))    # -10..+10 step 5 (5)
ELEVATION_CLAMP = (-45, 60)                          # keep poses physically plausible

# Low resolution during the search; the winning pose is re-rendered full-size.
MASK_SIZE = (256, 256)
SEARCH_RESOLUTION = (256, 256)


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
    # Explicit grids force a single-stage search; otherwise use coarse→fine.
    explicit = azimuths is not None or elevations is not None
    azimuths = azimuths or DEFAULT_AZIMUTHS
    elevations = elevations or DEFAULT_ELEVATIONS

    # Step 1: input silhouette mask. Normalise (tight-crop + aspect-preserving
    # resize) so pose scoring depends on SHAPE/rotation, not on how the object
    # happens to be framed/scaled in the photo vs the render.
    input_mask = extract_silhouette_mask(image_path, MASK_SIZE)
    input_mask_path = _save_mask(input_mask, output_dir, "input_mask")
    input_norm = _normalize_mask(input_mask)

    # Steps 2-4: scan candidates with a SINGLE reusable renderer (GLB loaded
    # once), computing masks in memory (no per-candidate file writes).
    best = None  # (similarity, iou, contour, az, el)
    evaluated = 0
    try:
        with PoseRenderer(glb_path, resolution=SEARCH_RESOLUTION) as pr:
            # Stage 1 — coarse grid over all sides.
            coarse = [(az, el) for el in elevations for az in azimuths]
            best, n = _scan(pr, input_norm, coarse)
            evaluated += n

            # Stage 2 — fine grid centred on the coarse winner (skipped when the
            # caller supplied an explicit grid, or coarse found nothing).
            if best is not None and not explicit:
                _, _, _, win_az, win_el = best
                fine = _fine_grid(win_az, win_el, seen=set(coarse))
                fine_best, n = _scan(pr, input_norm, fine)
                evaluated += n
                if fine_best is not None and (best is None or fine_best[0] > best[0]):
                    best = fine_best
    except Exception:
        logger.exception("PoseRenderer scan failed entirely")
        best = None
    finally:
        gc.collect()

    if best is None:
        logger.warning("pose estimation found no candidate (evaluated=%d); using fallback render", evaluated)
        # Could not scan any candidate — fall back to a single front render.
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

    similarity, iou, contour, az, el = best

    # Step 6: render ONLY the winning pose at full resolution and save it.
    aligned_path = render_glb_from_pose(
        glb_path, az, el, output_dir, resolution=resolution, suffix="aligned_render"
    )
    gc.collect()

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
        "candidates_evaluated": evaluated,
        "search_space": {"azimuths": azimuths, "elevations": elevations},
        "fallback": False,
    }


# ──────────────── Candidate scanning ────────────────


def _scan(pr, input_norm: np.ndarray, candidates):
    """
    Score a list of (azimuth, elevation) candidates against the input mask.

    Returns (best, evaluated) where best is (similarity, iou, contour, az, el)
    or None if no candidate could be rendered.
    """
    best = None
    evaluated = 0
    for az, el in candidates:
        try:
            render_mask = pr.mask_at(az, el)
            if render_mask.shape[:2] != MASK_SIZE:
                render_mask = cv2.resize(
                    render_mask, MASK_SIZE, interpolation=cv2.INTER_NEAREST
                )
        except Exception:
            logger.exception("pose candidate az=%s el=%s failed", az, el)
            continue

        evaluated += 1
        render_norm = _normalize_mask(render_mask)
        iou = _iou(input_norm, render_norm)
        contour = _contour_overlap(input_norm, render_norm)
        similarity = 0.7 * iou + 0.3 * contour

        if best is None or similarity > best[0]:
            best = (similarity, iou, contour, az, el)
    return best, evaluated


def _fine_grid(win_az: int, win_el: int, seen: set):
    """Dense (az, el) grid centred on the coarse winner, minus already-scanned
    poses, with azimuth wrapped to [0,360) and elevation clamped to plausible."""
    grid = []
    for d_el in FINE_ELEVATION_OFFSETS:
        el = int(np.clip(win_el + d_el, *ELEVATION_CLAMP))
        for d_az in FINE_AZIMUTH_OFFSETS:
            az = (win_az + d_az) % 360
            if (az, el) in seen:
                continue
            seen.add((az, el))
            grid.append((az, el))
    return grid


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Tight-crop the foreground to its bounding box, then scale it (preserving
    aspect ratio) onto a centred MASK_SIZE canvas. Removes scale + translation
    differences so pose scoring reflects SHAPE/orientation, not framing.
    """
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return mask
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = mask[y0:y1, x0:x1]

    tw, th = MASK_SIZE
    margin = 0.05
    box = int(min(tw, th) * (1 - 2 * margin))
    ch, cw = crop.shape
    scale = box / max(ch, cw)
    nh, nw = max(1, int(round(ch * scale))), max(1, int(round(cw * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((th, tw), dtype=mask.dtype)
    oy, ox = (th - nh) // 2, (tw - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


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
