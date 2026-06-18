"""
Camera Pose Estimator

Finds the camera viewpoint AND camera intrinsics (distance, FOV, image offset)
from which a rendered GLB best matches the silhouette of the background-removed
input image. The aligned render is reused by every downstream validator so the
photo and the model are compared from the same viewpoint.

Pipeline:
  1. Extract a binary silhouette mask from the input image.
  2. Hierarchical rotation search (45° → 10° → 2° → 0.5°) scored on SHAPE only
     (normalised masks), so orientation is decoupled from scale/translation.
  3. FOV + camera-distance + image-offset optimisation on RAW masks, so the
     render lands at the right size and position in the frame.
  4. A bounded joint refinement (scipy) that polishes all six parameters
     together against a combined alignment loss.
  5. Render only the winning pose/intrinsics at full resolution.

Everything runs through a SINGLE reusable PoseRenderer (GLB loaded once, masks
computed in memory) so memory stays bounded and no per-candidate files are
written.
"""

import os
import uuid
import gc
import logging
import numpy as np
import cv2

from services.glb_renderer import (
    render_glb_from_pose,
    extract_silhouette_mask,
    PoseRenderer,
    DEFAULT_FOV_DEG,
)

logger = logging.getLogger(__name__)

# ── Rotation search (Phase 1 + Phase 6: hierarchical refinement) ──
# Stage 1 — coarse grid over all sides.
STAGE1_AZIMUTHS = list(range(0, 360, 45))      # 0,45,...,315  (8)
STAGE1_ELEVATIONS = [-15, 0, 15, 30]           # (4) → 32 candidates
# Stages 2-4 — each searches a shrinking window around the previous winner.
# (azimuth step, azimuth half-window, elevation step, elevation half-window) °.
REFINE_STAGES = [
    (10.0, 40.0, 10.0, 20.0),
    (2.0, 10.0, 4.0, 8.0),
    (0.5, 2.0, 1.0, 2.0),
]
ELEVATION_CLAMP = (-45.0, 60.0)

# ── FOV search (Phase 4) ──
FOV_CANDIDATES = [35.0, 45.0, 55.0, 65.0, 75.0]

# ── Optimiser bounds / iteration caps ──
DIST_TOL = 0.02            # stop distance opt at <2% bbox-height error
MAX_DIST_ITERS = 6
MAX_OFFSET_ITERS = 4
JOINT_MAXITER = 30         # keep scipy bounded → bounded render count

# Low resolution during the search; the winning pose is re-rendered full-size.
MASK_SIZE = (256, 256)                 # (w, h)
MASK_HW = (MASK_SIZE[1], MASK_SIZE[0]) # (h, w) for shape comparisons
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
    Search viewpoint + intrinsics and return the best-aligned render.

    Preserves the original response keys (azimuth, elevation, iou,
    contour_overlap, confidence, aligned_render_*, input_mask_*,
    candidates_evaluated, search_space, fallback) and adds a "debug" block with
    the full alignment metrics.
    """
    explicit = azimuths is not None or elevations is not None

    # Step 1: input silhouette mask. Keep a RAW copy (real scale/position) for
    # scale/offset/FOV optimisation, and a NORMALISED copy (shape only) for the
    # rotation search.
    input_mask = extract_silhouette_mask(image_path, MASK_SIZE)
    input_mask_path = _save_mask(input_mask, output_dir, "input_mask")
    input_norm = _normalize_mask(input_mask)
    input_bbox = get_mask_bbox(input_mask)

    best = None
    evaluated = 0
    params = None  # dict of final camera params
    metrics = None

    try:
        with PoseRenderer(glb_path, resolution=SEARCH_RESOLUTION, fov_deg=DEFAULT_FOV_DEG) as pr:
            # ── Phase 6: hierarchical rotation search (shape-only scoring) ──
            best, evaluated = _rotation_search(
                pr, input_norm, explicit, azimuths, elevations
            )

            if best is not None and input_bbox is not None:
                az, el = best[4], best[5]
                # ── Phases 2+3+4: FOV / distance / offset on RAW masks ──
                params = _optimize_intrinsics(pr, input_mask, input_bbox, az, el)
                # ── Phase 7: bounded joint refinement of all 6 params ──
                params = _joint_refine(pr, input_mask, input_bbox, params)
                # Final metrics from the optimised render.
                final_mask = _render_mask(pr, **params)
                metrics = compute_alignment_metrics(input_mask, input_bbox, final_mask)
    except Exception:
        logger.exception("PoseRenderer scan failed entirely")
        best = None
    finally:
        gc.collect()

    if best is None or params is None or metrics is None:
        logger.warning(
            "pose estimation incomplete (evaluated=%d); using fallback render", evaluated
        )
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
            "candidates_evaluated": evaluated,
            "search_space": {"azimuths": STAGE1_AZIMUTHS, "elevations": STAGE1_ELEVATIONS},
            "fallback": True,
        }

    # ── Step 5: render ONLY the winning pose/intrinsics at full resolution ──
    aligned_path = render_glb_from_pose(
        glb_path,
        azimuth=params["azimuth"],
        elevation=params["elevation"],
        output_dir=output_dir,
        resolution=resolution,
        suffix="aligned_render",
        distance=params["distance"],
        fov_deg=params["fov_deg"],
        offset_x=params["offset_x"],
        offset_y=params["offset_y"],
    )
    gc.collect()

    final_score = _final_score(metrics)

    # ── Phase 8: debug metrics ──
    debug = {
        "azimuth": round(float(params["azimuth"]), 3),
        "elevation": round(float(params["elevation"]), 3),
        "distance": round(float(params["distance"]), 6),
        "fov": round(float(params["fov_deg"]), 2),
        "offset_x": round(float(params["offset_x"]), 6),
        "offset_y": round(float(params["offset_y"]), 6),
        "iou": round(float(metrics["iou"]), 4),
        "contour_overlap": round(float(metrics["contour"]), 4),
        "chamfer_similarity": round(float(metrics["chamfer"]), 4),
        "scale_error": round(float(metrics["scale_error"]), 4),
        "center_error": round(float(metrics["center_error"]), 4),
        "final_score": round(float(final_score), 4),
    }

    return {
        # Preserved API keys ──────────────────────────────
        "azimuth": round(float(params["azimuth"]), 2),
        "elevation": round(float(params["elevation"]), 2),
        "iou": round(float(metrics["iou"]), 4),
        "contour_overlap": round(float(metrics["contour"]), 4),
        "confidence": round(float(final_score), 4),
        "aligned_render_path": aligned_path,
        "aligned_render_url": f"/outputs/{os.path.basename(aligned_path)}",
        "input_mask_path": input_mask_path,
        "input_mask_url": f"/outputs/{os.path.basename(input_mask_path)}",
        "candidates_evaluated": evaluated,
        "search_space": {"azimuths": STAGE1_AZIMUTHS, "elevations": STAGE1_ELEVATIONS},
        "fallback": False,
        # New ──────────────────────────────────────────────
        "debug": debug,
    }


# ──────────────── Rotation search (Phase 1 + 6) ────────────────


def _rotation_search(pr, input_norm, explicit, azimuths, elevations):
    """Hierarchical az/el search scored on SHAPE only (normalised masks)."""
    evaluated = 0

    # Stage 1 — coarse grid (or caller-supplied explicit grid).
    az_grid = azimuths if azimuths is not None else STAGE1_AZIMUTHS
    el_grid = elevations if elevations is not None else STAGE1_ELEVATIONS
    coarse = [(float(az), float(el)) for el in el_grid for az in az_grid]
    seen = set(coarse)
    best, n = _scan(pr, input_norm, coarse)
    evaluated += n

    # Explicit grid → single-stage search (preserve old behaviour).
    if best is None or explicit:
        return best, evaluated

    # Stages 2-4 — shrinking windows around the running winner.
    for az_step, az_half, el_step, el_half in REFINE_STAGES:
        win_az, win_el = best[4], best[5]
        cand = _window(win_az, win_el, az_step, az_half, el_step, el_half, seen)
        if not cand:
            continue
        stage_best, n = _scan(pr, input_norm, cand)
        evaluated += n
        if stage_best is not None and stage_best[0] > best[0]:
            best = stage_best
    return best, evaluated


def _scan(pr, input_norm, candidates):
    """Score (az, el) candidates against the normalised input mask.

    Returns (best, evaluated); best = (score, iou, contour, chamfer, az, el)."""
    best = None
    evaluated = 0
    for az, el in candidates:
        try:
            render_mask = _render_mask(pr, azimuth=az, elevation=el)
        except Exception:
            logger.exception("pose candidate az=%s el=%s failed", az, el)
            continue
        evaluated += 1
        render_norm = _normalize_mask(render_mask)
        iou = _iou(input_norm, render_norm)
        contour = _contour_overlap(input_norm, render_norm)
        chamfer = chamfer_similarity(input_norm, render_norm)
        score = 0.5 * iou + 0.2 * contour + 0.3 * chamfer  # Phase 5 metric
        if best is None or score > best[0]:
            best = (score, iou, contour, chamfer, az, el)
    return best, evaluated


def _window(win_az, win_el, az_step, az_half, el_step, el_half, seen):
    """Candidate grid in a window around (win_az, win_el), wrapped/clamped and
    de-duplicated against already-scanned poses."""
    grid = []
    for el in _arange(win_el - el_half, win_el + el_half, el_step):
        el = float(np.clip(el, *ELEVATION_CLAMP))
        for az in _arange(win_az - az_half, win_az + az_half, az_step):
            az = float(round(az % 360, 4))
            key = (az, round(el, 4))
            if key in seen:
                continue
            seen.add(key)
            grid.append((az, el))
    return grid


def _arange(lo, hi, step):
    n = int(round((hi - lo) / step))
    return [lo + i * step for i in range(n + 1)]


# ──────────────── Intrinsics optimisation (Phases 2, 3, 4) ────────────────


def _optimize_intrinsics(pr, input_mask, input_bbox, az, el):
    """Search FOV candidates; for each, optimise distance then offset, and keep
    the FOV that minimises the alignment loss. Returns a params dict."""
    best = None
    for fov in FOV_CANDIDATES:
        distance = optimize_camera_distance(pr, input_bbox, az, el, fov, pr.default_distance)
        offset_x, offset_y = optimize_camera_offset(pr, input_bbox, az, el, distance, fov)
        mask = _render_mask(pr, azimuth=az, elevation=el, distance=distance,
                            fov_deg=fov, offset_x=offset_x, offset_y=offset_y)
        m = compute_alignment_metrics(input_mask, input_bbox, mask)
        loss = _alignment_loss(m)
        if best is None or loss < best[0]:
            best = (loss, {
                "azimuth": az, "elevation": el, "distance": distance,
                "fov_deg": fov, "offset_x": offset_x, "offset_y": offset_y,
            })
    return best[1]


def optimize_camera_distance(pr, input_bbox, az, el, fov, distance0):
    """Iteratively scale camera distance until the rendered bbox height matches
    the input bbox height within DIST_TOL (Phase 2)."""
    input_h = max(input_bbox[3], 1)
    distance = float(distance0)
    lo, hi = 0.2 * pr.default_distance, 6.0 * pr.default_distance
    for _ in range(MAX_DIST_ITERS):
        mask = _render_mask(pr, azimuth=az, elevation=el, distance=distance, fov_deg=fov)
        bb = get_mask_bbox(mask)
        if bb is None or bb[3] <= 0:
            break
        render_h = bb[3]
        err = abs(input_h - render_h) / input_h
        if err < DIST_TOL:
            break
        scale_ratio = input_h / render_h          # >1 ⇒ render too small ⇒ move closer
        distance = float(np.clip(distance / scale_ratio, lo, hi))
    return distance


def optimize_camera_offset(pr, input_bbox, az, el, distance, fov):
    """Pan the camera so the rendered object's centre matches the input centre
    (Phase 3). Sign/scale of the world→pixel mapping is recovered numerically
    (finite differences) so it is convention-agnostic, then applied by Newton
    steps."""
    icx = input_bbox[0] + input_bbox[2] / 2.0
    icy = input_bbox[1] + input_bbox[3] / 2.0

    def center(ox, oy):
        mask = _render_mask(pr, azimuth=az, elevation=el, distance=distance,
                            fov_deg=fov, offset_x=ox, offset_y=oy)
        bb = get_mask_bbox(mask)
        if bb is None:
            return None
        return bb[0] + bb[2] / 2.0, bb[1] + bb[3] / 2.0

    eps = max(distance * 0.02, 1e-4)
    base = center(0.0, 0.0)
    if base is None:
        return 0.0, 0.0
    cx_e = center(eps, 0.0)
    cy_e = center(0.0, eps)
    # px shift per unit world-offset along each camera axis.
    sx = (cx_e[0] - base[0]) / eps if cx_e is not None else 0.0
    sy = (cy_e[1] - base[1]) / eps if cy_e is not None else 0.0

    ox = oy = 0.0
    lim = 2.0 * pr.scene_size
    for _ in range(MAX_OFFSET_ITERS):
        c = center(ox, oy)
        if c is None:
            break
        ex, ey = icx - c[0], icy - c[1]
        if abs(ex) < 0.5 and abs(ey) < 0.5:
            break
        if abs(sx) > 1e-6:
            ox = float(np.clip(ox + ex / sx, -lim, lim))
        if abs(sy) > 1e-6:
            oy = float(np.clip(oy + ey / sy, -lim, lim))
    return ox, oy


def optimize_fov(pr, input_mask, input_bbox, az, el):
    """Phase 4 entry point: choose the FOV minimising alignment loss. (The full
    pipeline uses _optimize_intrinsics which folds this together with distance
    and offset; exposed separately per the spec.)"""
    return _optimize_intrinsics(pr, input_mask, input_bbox, az, el)["fov_deg"]


# ──────────────── Joint refinement (Phase 7) ────────────────


def _joint_refine(pr, input_mask, input_bbox, params):
    """Bounded joint optimisation of (az, el, distance, fov, offset_x, offset_y)
    around the staged winner using scipy. Falls back to the seed on any error.
    """
    try:
        from scipy.optimize import minimize
    except Exception:
        logger.info("scipy unavailable; skipping joint refinement")
        return params

    az0, el0 = params["azimuth"], params["elevation"]
    d0, fov0 = params["distance"], params["fov_deg"]
    ox0, oy0 = params["offset_x"], params["offset_y"]
    lim = 2.0 * pr.scene_size

    x0 = np.array([az0, el0, d0, fov0, ox0, oy0], dtype=float)
    bounds = [
        (az0 - 5, az0 + 5),
        (float(np.clip(el0 - 5, *ELEVATION_CLAMP)), float(np.clip(el0 + 5, *ELEVATION_CLAMP))),
        (0.6 * d0, 1.6 * d0),
        (FOV_CANDIDATES[0], FOV_CANDIDATES[-1]),
        (ox0 - 0.3 * pr.scene_size, ox0 + 0.3 * pr.scene_size),
        (oy0 - 0.3 * pr.scene_size, oy0 + 0.3 * pr.scene_size),
    ]

    def loss(x):
        try:
            mask = _render_mask(
                pr, azimuth=x[0], elevation=x[1], distance=x[2],
                fov_deg=float(np.clip(x[3], FOV_CANDIDATES[0], FOV_CANDIDATES[-1])),
                offset_x=float(np.clip(x[4], -lim, lim)),
                offset_y=float(np.clip(x[5], -lim, lim)),
            )
            return _alignment_loss(compute_alignment_metrics(input_mask, input_bbox, mask))
        except Exception:
            return 10.0

    seed_loss = loss(x0)
    try:
        res = minimize(
            loss, x0, method="Powell", bounds=bounds,
            options={"maxiter": JOINT_MAXITER, "maxfev": 90, "xtol": 1e-3, "ftol": 1e-3},
        )
    except Exception:
        logger.exception("joint refinement failed; keeping staged params")
        return params

    if res.success or res.fun < seed_loss:
        x = res.x
        return {
            "azimuth": float(x[0]),
            "elevation": float(np.clip(x[1], *ELEVATION_CLAMP)),
            "distance": float(x[2]),
            "fov_deg": float(np.clip(x[3], FOV_CANDIDATES[0], FOV_CANDIDATES[-1])),
            "offset_x": float(np.clip(x[4], -lim, lim)),
            "offset_y": float(np.clip(x[5], -lim, lim)),
        }
    return params


# ──────────────── Bounding box / error helpers (Phases 2, 3) ────────────────


def get_mask_bbox(mask: np.ndarray):
    """Return (x0, y0, w, h) of the foreground bbox, or None if empty."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def compute_scale_error(input_bbox, render_bbox) -> float:
    """Relative bbox-height error |input_h - render_h| / input_h."""
    if input_bbox is None or render_bbox is None:
        return 1.0
    input_h = max(input_bbox[3], 1)
    return abs(input_h - render_bbox[3]) / input_h


def compute_center_error(input_bbox, render_bbox) -> float:
    """Centre offset between bboxes, normalised by the image diagonal."""
    if input_bbox is None or render_bbox is None:
        return 1.0
    icx = input_bbox[0] + input_bbox[2] / 2.0
    icy = input_bbox[1] + input_bbox[3] / 2.0
    rcx = render_bbox[0] + render_bbox[2] / 2.0
    rcy = render_bbox[1] + render_bbox[3] / 2.0
    diag = float(np.hypot(*MASK_HW))
    return float(np.hypot(icx - rcx, icy - rcy) / diag)


# ──────────────── Similarity metrics (Phase 5) ────────────────


def compute_alignment_metrics(input_mask, input_bbox, render_mask) -> dict:
    """Full RAW-mask alignment metrics (no normalisation): real scale + position
    matter here, unlike the rotation search."""
    render_bbox = get_mask_bbox(render_mask)
    return {
        "iou": _iou(input_mask, render_mask),
        "contour": _contour_overlap(input_mask, render_mask),
        "chamfer": chamfer_similarity(input_mask, render_mask),
        "scale_error": compute_scale_error(input_bbox, render_bbox),
        "center_error": compute_center_error(input_bbox, render_bbox),
    }


def _final_score(m: dict) -> float:
    return 0.5 * m["iou"] + 0.2 * m["contour"] + 0.3 * m["chamfer"]


def _alignment_loss(m: dict) -> float:
    """Phase 7 objective: low = good. chamfer_distance enters as (1 - chamfer
    similarity), a bounded monotonic proxy for the symmetric contour distance."""
    return (1.0 - m["iou"]) + (1.0 - m["chamfer"]) + m["scale_error"] + m["center_error"]


def chamfer_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Normalised symmetric Chamfer similarity between the two silhouettes' contours.

    1. Extract contour bands.
    2. Distance-transform each (distance to nearest contour pixel).
    3. Symmetric mean contour-to-contour distance.
    4. Map to a [0,1] similarity via an exponential of the diagonal-normalised
       distance (0 = far apart, 1 = perfectly coincident edges).
    """
    edges_a = _contour_band(mask_a, thickness=1)
    edges_b = _contour_band(mask_b, thickness=1)
    if edges_a.sum() == 0 or edges_b.sum() == 0:
        return 0.0

    inv_a = np.where(edges_a, 0, 255).astype(np.uint8)
    inv_b = np.where(edges_b, 0, 255).astype(np.uint8)
    dt_a = cv2.distanceTransform(inv_a, cv2.DIST_L2, 3)
    dt_b = cv2.distanceTransform(inv_b, cv2.DIST_L2, 3)

    a_to_b = float(dt_b[edges_a].mean())
    b_to_a = float(dt_a[edges_b].mean())
    chamfer = (a_to_b + b_to_a) / 2.0

    diag = float(np.hypot(*MASK_HW))
    return float(np.exp(-chamfer / (0.05 * diag)))


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def _contour_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Symmetric contour-band overlap: dilate each contour into a thin band and
    measure how much of one band falls on the other."""
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


# ──────────────── Mask utilities ────────────────


def _render_mask(pr, azimuth, elevation, distance=None, fov_deg=None,
                 offset_x=0.0, offset_y=0.0) -> np.ndarray:
    """Render a silhouette mask at MASK_SIZE through the reusable renderer."""
    mask = pr.mask_at(azimuth, elevation, distance=distance, fov_deg=fov_deg,
                      offset_x=offset_x, offset_y=offset_y)
    if mask.shape[:2] != MASK_HW:
        mask = cv2.resize(mask, MASK_SIZE, interpolation=cv2.INTER_NEAREST)
    return mask


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Tight-crop the foreground to its bounding box, then scale it (preserving
    aspect ratio) onto a centred MASK_SIZE canvas. Removes scale + translation
    so ROTATION scoring reflects SHAPE only. (Scale/translation are recovered
    separately by the intrinsics optimisers.)
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


def _save_mask(mask: np.ndarray, output_dir: str, suffix: str) -> str:
    path = os.path.join(output_dir, f"{uuid.uuid4()}_{suffix}.png")
    cv2.imwrite(path, mask)
    return path
