"""
Camera Pose Estimator

Estimates the camera viewpoint + intrinsics from which a rendered GLB best
matches a background-removed product photo. The aligned render is reused by
every downstream validator so the photo and the model are compared from the
same viewing angle.

WHY THIS IS MULTI-STAGE
-----------------------
A silhouette alone cannot disambiguate symmetric furniture: the outline of a
chair's FRONT is nearly identical to its BACK, so pure-IoU search happily
returns an ~88%-confident rear view. Silhouette is therefore used ONLY for
coarse candidate generation; the final pose is chosen by APPEARANCE.

Pipeline:
  Stage 1  Silhouette coarse scan over a full az/el grid (shape-only scoring).
  Stage 2  Top-K selection with angular non-max-suppression (keeps front AND
           back), then a small local silhouette refinement per candidate.
  Stage 3  Textured RGB render of each Top-K pose (in-memory, one GLB load).
  Stage 4  Visual scoring: edge similarity (Canny + chamfer + overlap),
           internal-feature score (edges INSIDE the silhouette — cushions,
           armrests, legs), SSIM, colour histogram.
  Stage 5  Final ranking:
               final_score = 0.3*silhouette + 0.4*edge + 0.3*visual
           Winner + runner-up recorded; intrinsics (distance/FOV/offset) and a
           bounded joint refinement are then optimised on the winner.

Everything runs through a SINGLE reusable PoseRenderer (one GLB load, masks &
RGB rendered in memory). If textured RGB rendering is unavailable on the host,
the system degrades gracefully to silhouette-only ranking.
"""

import os
import uuid
import gc
import logging
import numpy as np
import cv2

from skimage.metrics import structural_similarity as ssim

from services.glb_renderer import (
    render_glb_from_pose,
    extract_silhouette_mask,
    PoseRenderer,
    DEFAULT_FOV_DEG,
)

logger = logging.getLogger(__name__)

# ── Rotation search (coarse) ──
# Finer azimuth/elevation grid (Fix 4) so the coarse scan lands close enough for
# local refinement + the joint optimiser to drive silhouette IoU up. 20° azimuth
# steps × 6 elevations = 108 candidates (was 30° × 4 = 48).
STAGE1_AZIMUTHS = list(range(0, 360, 20))      # 0,20,...,340 (18)
STAGE1_ELEVATIONS = [-20, -10, 0, 15, 30, 45]  # opposite sides well sampled
ELEVATION_CLAMP = (-45.0, 60.0)

# Per-candidate local refinement window (degrees) — two scales for a finer
# silhouette-IoU search around each Top-K candidate before the joint optimiser.
REFINE_AZ = [-6.0, -3.0, -1.5, 0.0, 1.5, 3.0, 6.0]
REFINE_EL = [-6.0, -3.0, -1.5, 0.0, 1.5, 3.0, 6.0]

# ── Top-K appearance re-ranking ──
TOP_K = 8
MIN_AZ_SEP = 30.0      # angular non-max-suppression: keep viewpoints ≥30° apart
MIN_EL_SEP = 20.0
# Final-score weights (Phase 7).
W_SIL, W_EDGE, W_VIS = 0.3, 0.4, 0.3
# Visual sub-score weights.
W_SSIM, W_HIST, W_INTERNAL = 0.5, 0.2, 0.3

# ── FOV search ──
FOV_CANDIDATES = [35.0, 45.0, 55.0, 65.0, 75.0]

# ── Optimiser bounds / iteration caps ──
DIST_TOL = 0.02
MAX_DIST_ITERS = 6
MAX_OFFSET_ITERS = 4
JOINT_MAXITER = 30

MASK_SIZE = (256, 256)                 # (w, h)
MASK_HW = (MASK_SIZE[1], MASK_SIZE[0]) # (h, w)
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
    Multi-stage pose estimation. Preserves the original response keys and adds a
    "debug" block with the Top-K candidate scores, winner, and runner-up.
    """
    explicit = azimuths is not None or elevations is not None

    # Step 1: input silhouette + a colour copy of the source (composited over
    # white, so it matches the over-white textured renders for SSIM/hist).
    input_mask = extract_silhouette_mask(image_path, MASK_SIZE)
    input_mask_path = _save_mask(input_mask, output_dir, "input_mask")
    input_norm = _normalize_mask(input_mask)
    input_bbox = get_mask_bbox(input_mask)
    src_bgr = _load_source_bgr(image_path, MASK_SIZE)
    src_gray_n, src_mask_n = _normalize_gray(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY), input_mask)
    src_bgr_n, _ = _normalize_color(src_bgr, input_mask)

    params = None
    metrics = None
    evaluated = 0
    ranked = []

    try:
        with PoseRenderer(glb_path, resolution=SEARCH_RESOLUTION, fov_deg=DEFAULT_FOV_DEG) as pr:
            # ── Stage 1: coarse silhouette scan (all candidates) ──
            az_grid = azimuths if azimuths is not None else STAGE1_AZIMUTHS
            el_grid = elevations if elevations is not None else STAGE1_ELEVATIONS
            grid = [(float(az), float(el)) for el in el_grid for az in az_grid]
            cands, evaluated = _coarse_scan(pr, input_norm, grid)

            if cands:
                # ── Stage 2: Top-K (angular NMS) + local refinement ──
                topk = _select_topk(cands)
                if not explicit:
                    for c in topk:
                        _refine_candidate(pr, input_norm, c)

                # ── Stage 3+4: textured RGB render + visual scoring ──
                ranked = _visual_rerank(
                    pr, src_gray_n, src_mask_n, src_bgr_n, topk, output_dir
                )

                # ── Stage 5: final ranking ──
                ranked.sort(key=lambda c: c["final_score"], reverse=True)
                winner = ranked[0]

                # Intrinsics + joint refinement on the winner only.
                if input_bbox is not None:
                    params = _optimize_intrinsics(
                        pr, input_mask, input_bbox, winner["azimuth"], winner["elevation"]
                    )
                    params = _joint_refine(pr, input_mask, input_bbox, params)
                    final_mask = _render_mask(pr, **params)
                    metrics = compute_alignment_metrics(input_mask, input_bbox, final_mask)
    except Exception:
        logger.exception("pose estimation failed")
        params = None

    if params is None or metrics is None:
        logger.warning("pose estimation incomplete (evaluated=%d); fallback render", evaluated)
        aligned = render_glb_from_pose(glb_path, 0, 0, output_dir, resolution=resolution, suffix="aligned")
        return {
            "azimuth": 0, "elevation": 0, "iou": 0.0, "contour_overlap": 0.0,
            "confidence": 0.0,
            "aligned_render_path": aligned,
            "aligned_render_url": f"/outputs/{os.path.basename(aligned)}",
            "input_mask_path": input_mask_path,
            "input_mask_url": f"/outputs/{os.path.basename(input_mask_path)}",
            "candidates_evaluated": evaluated,
            "search_space": {"azimuths": STAGE1_AZIMUTHS, "elevations": STAGE1_ELEVATIONS},
            "fallback": True,
        }

    # ── Final full-resolution render with optimised params ──
    aligned_path = render_glb_from_pose(
        glb_path, azimuth=params["azimuth"], elevation=params["elevation"],
        output_dir=output_dir, resolution=resolution, suffix="aligned_render",
        distance=params["distance"], fov_deg=params["fov_deg"],
        offset_x=params["offset_x"], offset_y=params["offset_y"],
    )
    gc.collect()

    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    visual_used = winner.get("visual_similarity") is not None

    # ── Phase 8: debug output ──
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
        "silhouette_score": round(float(winner["silhouette_score"]), 4),
        "edge_similarity": _r(winner.get("edge_similarity")),
        "visual_similarity": _r(winner.get("visual_similarity")),
        "final_score": round(float(winner["final_score"]), 4),
        "visual_reranking": visual_used,
        "weights": {"silhouette": W_SIL, "edge": W_EDGE, "visual": W_VIS},
        "topk": [_candidate_debug(c) for c in ranked],
        "runner_up": _candidate_debug(runner_up) if runner_up else None,
    }

    # ── Phase 9: optional debug grid ──
    if os.environ.get("POSE_DEBUG_GRID"):
        grid_url = _save_debug_grid(ranked, output_dir)
        if grid_url:
            debug["debug_grid_url"] = grid_url

    return {
        "azimuth": round(float(params["azimuth"]), 2),
        "elevation": round(float(params["elevation"]), 2),
        "iou": round(float(metrics["iou"]), 4),
        "contour_overlap": round(float(metrics["contour"]), 4),
        "confidence": round(float(winner["final_score"]), 4),
        "aligned_render_path": aligned_path,
        "aligned_render_url": f"/outputs/{os.path.basename(aligned_path)}",
        "input_mask_path": input_mask_path,
        "input_mask_url": f"/outputs/{os.path.basename(input_mask_path)}",
        "candidates_evaluated": evaluated,
        "search_space": {"azimuths": STAGE1_AZIMUTHS, "elevations": STAGE1_ELEVATIONS},
        "fallback": False,
        "debug": debug,
    }


# ──────────────── Stage 1: coarse silhouette scan ────────────────


def _coarse_scan(pr, input_norm, grid):
    """Score every (az, el) on shape-only silhouette metrics. Returns
    (list_of_candidate_dicts, evaluated)."""
    cands = []
    evaluated = 0
    for az, el in grid:
        try:
            mask = _render_mask(pr, azimuth=az, elevation=el)
        except Exception:
            logger.exception("candidate az=%s el=%s failed", az, el)
            continue
        evaluated += 1
        cands.append(_score_silhouette(input_norm, mask, az, el))
    return cands, evaluated


def _score_silhouette(input_norm, render_mask, az, el) -> dict:
    rn = _normalize_mask(render_mask)
    iou = _iou(input_norm, rn)
    contour = _contour_overlap(input_norm, rn)
    chamfer = chamfer_similarity(input_norm, rn)
    return {
        "azimuth": az, "elevation": el,
        "iou": iou, "contour_overlap": contour, "chamfer_score": chamfer,
        "silhouette_score": 0.5 * iou + 0.2 * contour + 0.3 * chamfer,
    }


# ──────────────── Stage 2: Top-K + local refinement ────────────────


def _select_topk(cands):
    """Pick TOP_K candidates by silhouette score with angular non-max
    suppression so distinct viewpoints (notably front vs back) all survive."""
    ordered = sorted(cands, key=lambda c: c["silhouette_score"], reverse=True)
    picked, leftover = [], []
    for c in ordered:
        if any(_ang_close(c, p) for p in picked):
            leftover.append(c)
            continue
        picked.append(c)
        if len(picked) >= TOP_K:
            break
    # Backfill from suppressed candidates if NMS left us short.
    for c in leftover:
        if len(picked) >= TOP_K:
            break
        picked.append(c)
    return picked[:TOP_K]


def _ang_close(a, b) -> bool:
    daz = abs((a["azimuth"] - b["azimuth"] + 180) % 360 - 180)
    return daz < MIN_AZ_SEP and abs(a["elevation"] - b["elevation"]) < MIN_EL_SEP


def _refine_candidate(pr, input_norm, c):
    """Small local silhouette search around a candidate; updates it in place."""
    best = c
    for d_el in REFINE_EL:
        el = float(np.clip(c["elevation"] + d_el, *ELEVATION_CLAMP))
        for d_az in REFINE_AZ:
            az = (c["azimuth"] + d_az) % 360
            try:
                mask = _render_mask(pr, azimuth=az, elevation=el)
            except Exception:
                continue
            s = _score_silhouette(input_norm, mask, az, el)
            if s["silhouette_score"] > best["silhouette_score"]:
                best = s
    c.update(best)


# ──────────────── Stage 3+4: RGB render + visual scoring ────────────────


def _visual_rerank(pr, src_gray_n, src_mask_n, src_bgr_n, candidates, output_dir):
    """Render each candidate's textured RGB and compute appearance scores.

    Degrades to silhouette-only ranking if textured RGB is unavailable."""
    for c in candidates:
        rgb, mask = pr.rgb_at(c["azimuth"], c["elevation"])
        if rgb is None or mask is None or mask.sum() == 0:
            # Visual path unavailable → rank by silhouette alone.
            c["edge_similarity"] = None
            c["visual_similarity"] = None
            c["final_score"] = c["silhouette_score"]
            c["_rgb"] = None
            continue

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray_n, mask_n = _normalize_gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), mask)
        bgr_n, _ = _normalize_color(bgr, mask)

        edge = edge_similarity(src_gray_n, gray_n)
        internal = internal_feature_score(src_gray_n, src_mask_n, gray_n, mask_n)
        ssim_v = _ssim01(src_gray_n, gray_n)
        hist = histogram_similarity(src_bgr_n, src_mask_n, bgr_n, mask_n)

        visual = W_SSIM * ssim_v + W_HIST * hist + W_INTERNAL * internal
        c["edge_similarity"] = float(edge)
        c["ssim"] = float(ssim_v)
        c["histogram"] = float(hist)
        c["internal_feature_score"] = float(internal)
        c["visual_similarity"] = float(visual)
        c["final_score"] = float(
            W_SIL * c["silhouette_score"] + W_EDGE * edge + W_VIS * visual
        )
        c["_rgb"] = bgr   # kept (small) for the optional debug grid
    return candidates


def edge_similarity(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Phase 3: Canny edges of both images compared by symmetric chamfer + band
    overlap. Captures armrests / cushion outlines / visible side geometry that
    the outer silhouette misses."""
    ea = cv2.Canny(gray_a, 50, 150) > 0
    eb = cv2.Canny(gray_b, 50, 150) > 0
    return _edge_map_similarity(ea, eb)


def internal_feature_score(gray_a, mask_a, gray_b, mask_b) -> float:
    """Phase 6: same as edge_similarity but restricted to edges INSIDE the
    silhouette (interior), where front/back differences live (cushion, seat,
    legs vs a flat backrest)."""
    k = np.ones((7, 7), np.uint8)
    int_a = cv2.erode(mask_a, k) > 0
    int_b = cv2.erode(mask_b, k) > 0
    ea = (cv2.Canny(gray_a, 50, 150) > 0) & int_a
    eb = (cv2.Canny(gray_b, 50, 150) > 0) & int_b
    return _edge_map_similarity(ea, eb)


def _edge_map_similarity(ea: np.ndarray, eb: np.ndarray) -> float:
    """Combine symmetric chamfer similarity and dilated-band overlap of two
    boolean edge maps into a [0,1] score."""
    if ea.sum() == 0 or eb.sum() == 0:
        return 0.0
    inv_a = np.where(ea, 0, 255).astype(np.uint8)
    inv_b = np.where(eb, 0, 255).astype(np.uint8)
    dt_a = cv2.distanceTransform(inv_a, cv2.DIST_L2, 3)
    dt_b = cv2.distanceTransform(inv_b, cv2.DIST_L2, 3)
    chamfer = (float(dt_b[ea].mean()) + float(dt_a[eb].mean())) / 2.0
    diag = float(np.hypot(*MASK_HW))
    chamfer_sim = float(np.exp(-chamfer / (0.05 * diag)))

    k = np.ones((3, 3), np.uint8)
    da = cv2.dilate(ea.astype(np.uint8), k) > 0
    db = cv2.dilate(eb.astype(np.uint8), k) > 0
    union = np.logical_or(da, db).sum()
    overlap = float(np.logical_and(da, db).sum() / union) if union else 0.0
    return 0.5 * chamfer_sim + 0.5 * overlap


def histogram_similarity(bgr_a, mask_a, bgr_b, mask_b) -> float:
    """HSV hue+saturation histogram correlation over the foreground, mapped to
    [0,1]. A weak but useful cue (e.g. wood vs upholstery distribution)."""
    m_a = (mask_a > 0).astype(np.uint8)
    m_b = (mask_b > 0).astype(np.uint8)
    hsv_a = cv2.cvtColor(bgr_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(bgr_b, cv2.COLOR_BGR2HSV)
    vals = []
    for ch, bins, rng in ((0, 32, [0, 180]), (1, 32, [0, 256])):
        ha = cv2.calcHist([hsv_a], [ch], m_a, [bins], rng)
        hb = cv2.calcHist([hsv_b], [ch], m_b, [bins], rng)
        cv2.normalize(ha, ha)
        cv2.normalize(hb, hb)
        vals.append(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
    corr = float(np.mean(vals)) if vals else 0.0
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


def _ssim01(gray_a, gray_b) -> float:
    try:
        v, _ = ssim(gray_a, gray_b, full=True, data_range=255)
        return float(np.clip(v, 0.0, 1.0))
    except Exception:
        return 0.0


# ──────────────── Intrinsics optimisation (distance / offset / FOV) ────────────────


def _optimize_intrinsics(pr, input_mask, input_bbox, az, el):
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
    input_h = max(input_bbox[3], 1)
    distance = float(distance0)
    lo, hi = 0.2 * pr.default_distance, 6.0 * pr.default_distance
    for _ in range(MAX_DIST_ITERS):
        mask = _render_mask(pr, azimuth=az, elevation=el, distance=distance, fov_deg=fov)
        bb = get_mask_bbox(mask)
        if bb is None or bb[3] <= 0:
            break
        err = abs(input_h - bb[3]) / input_h
        if err < DIST_TOL:
            break
        distance = float(np.clip(distance / (input_h / bb[3]), lo, hi))
    return distance


def optimize_camera_offset(pr, input_bbox, az, el, distance, fov):
    icx = input_bbox[0] + input_bbox[2] / 2.0
    icy = input_bbox[1] + input_bbox[3] / 2.0

    def center(ox, oy):
        mask = _render_mask(pr, azimuth=az, elevation=el, distance=distance,
                            fov_deg=fov, offset_x=ox, offset_y=oy)
        bb = get_mask_bbox(mask)
        return None if bb is None else (bb[0] + bb[2] / 2.0, bb[1] + bb[3] / 2.0)

    eps = max(distance * 0.02, 1e-4)
    base = center(0.0, 0.0)
    if base is None:
        return 0.0, 0.0
    cx_e = center(eps, 0.0)
    cy_e = center(0.0, eps)
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
    return _optimize_intrinsics(pr, input_mask, input_bbox, az, el)["fov_deg"]


def _joint_refine(pr, input_mask, input_bbox, params):
    try:
        from scipy.optimize import minimize
    except Exception:
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
        res = minimize(loss, x0, method="Powell", bounds=bounds,
                       options={"maxiter": JOINT_MAXITER, "maxfev": 90, "xtol": 1e-3, "ftol": 1e-3})
    except Exception:
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


# ──────────────── Bounding box / error helpers ────────────────


def get_mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def compute_scale_error(input_bbox, render_bbox) -> float:
    if input_bbox is None or render_bbox is None:
        return 1.0
    input_h = max(input_bbox[3], 1)
    return abs(input_h - render_bbox[3]) / input_h


def compute_center_error(input_bbox, render_bbox) -> float:
    if input_bbox is None or render_bbox is None:
        return 1.0
    icx = input_bbox[0] + input_bbox[2] / 2.0
    icy = input_bbox[1] + input_bbox[3] / 2.0
    rcx = render_bbox[0] + render_bbox[2] / 2.0
    rcy = render_bbox[1] + render_bbox[3] / 2.0
    diag = float(np.hypot(*MASK_HW))
    return float(np.hypot(icx - rcx, icy - rcy) / diag)


# ──────────────── Silhouette similarity metrics ────────────────


def compute_alignment_metrics(input_mask, input_bbox, render_mask) -> dict:
    render_bbox = get_mask_bbox(render_mask)
    return {
        "iou": _iou(input_mask, render_mask),
        "contour": _contour_overlap(input_mask, render_mask),
        "chamfer": chamfer_similarity(input_mask, render_mask),
        "scale_error": compute_scale_error(input_bbox, render_bbox),
        "center_error": compute_center_error(input_bbox, render_bbox),
    }


def _alignment_loss(m: dict) -> float:
    return (1.0 - m["iou"]) + (1.0 - m["chamfer"]) + m["scale_error"] + m["center_error"]


def chamfer_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    edges_a = _contour_band(mask_a, thickness=1)
    edges_b = _contour_band(mask_b, thickness=1)
    if edges_a.sum() == 0 or edges_b.sum() == 0:
        return 0.0
    inv_a = np.where(edges_a, 0, 255).astype(np.uint8)
    inv_b = np.where(edges_b, 0, 255).astype(np.uint8)
    dt_a = cv2.distanceTransform(inv_a, cv2.DIST_L2, 3)
    dt_b = cv2.distanceTransform(inv_b, cv2.DIST_L2, 3)
    chamfer = (float(dt_b[edges_a].mean()) + float(dt_a[edges_b].mean())) / 2.0
    diag = float(np.hypot(*MASK_HW))
    return float(np.exp(-chamfer / (0.05 * diag)))


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _contour_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
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


# ──────────────── Image / mask utilities ────────────────


def _render_mask(pr, azimuth, elevation, distance=None, fov_deg=None,
                 offset_x=0.0, offset_y=0.0) -> np.ndarray:
    mask = pr.mask_at(azimuth, elevation, distance=distance, fov_deg=fov_deg,
                      offset_x=offset_x, offset_y=offset_y)
    if mask.shape[:2] != MASK_HW:
        mask = cv2.resize(mask, MASK_SIZE, interpolation=cv2.INTER_NEAREST)
    return mask


def _load_source_bgr(image_path: str, size: tuple) -> np.ndarray:
    """Load the source image as BGR, alpha composited over white, resized."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[-1] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    """Tight-crop foreground to its bbox and scale (aspect-preserving) onto a
    centred canvas — removes scale/translation so ROTATION scoring is shape-only."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return mask
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = mask[y0:y1, x0:x1]
    tw, th = MASK_SIZE
    box = int(min(tw, th) * 0.9)
    ch, cw = crop.shape
    scale = box / max(ch, cw)
    nh, nw = max(1, int(round(ch * scale))), max(1, int(round(cw * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((th, tw), dtype=mask.dtype)
    oy, ox = (th - nh) // 2, (tw - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def _normalize_gray(gray: np.ndarray, mask: np.ndarray):
    """Crop the masked foreground to its bbox and centre-scale it onto a WHITE
    canvas (matching the over-white renders). Returns (gray_canvas, mask_canvas)
    so appearance is compared scale/translation-invariant."""
    ys, xs = np.where(mask > 0)
    tw, th = MASK_SIZE
    if ys.size == 0:
        return np.full((th, tw), 255, np.uint8), np.zeros((th, tw), np.uint8)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    g = gray[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1]
    box = int(min(tw, th) * 0.9)
    ch, cw = g.shape
    scale = box / max(ch, cw)
    nh, nw = max(1, int(round(ch * scale))), max(1, int(round(cw * scale)))
    g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
    gc_ = np.full((th, tw), 255, np.uint8)
    mc = np.zeros((th, tw), np.uint8)
    oy, ox = (th - nh) // 2, (tw - nw) // 2
    gc_[oy:oy + nh, ox:ox + nw] = g
    mc[oy:oy + nh, ox:ox + nw] = m
    # Force background to white so only foreground texture is compared.
    gc_[mc == 0] = 255
    return gc_, mc


def _normalize_color(bgr: np.ndarray, mask: np.ndarray):
    """Colour version of _normalize_gray (white background, centred bbox)."""
    ys, xs = np.where(mask > 0)
    tw, th = MASK_SIZE
    if ys.size == 0:
        return np.full((th, tw, 3), 255, np.uint8), np.zeros((th, tw), np.uint8)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    b = bgr[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1]
    box = int(min(tw, th) * 0.9)
    ch, cw = b.shape[:2]
    scale = box / max(ch, cw)
    nh, nw = max(1, int(round(ch * scale))), max(1, int(round(cw * scale)))
    b = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
    bc = np.full((th, tw, 3), 255, np.uint8)
    mc = np.zeros((th, tw), np.uint8)
    oy, ox = (th - nh) // 2, (tw - nw) // 2
    bc[oy:oy + nh, ox:ox + nw] = b
    mc[oy:oy + nh, ox:ox + nw] = m
    bc[mc == 0] = 255
    return bc, mc


# ──────────────── Debug helpers (Phase 8/9) ────────────────


def _r(x):
    return round(float(x), 4) if x is not None else None


def _candidate_debug(c: dict) -> dict:
    return {
        "azimuth": round(float(c["azimuth"]), 2),
        "elevation": round(float(c["elevation"]), 2),
        "iou": _r(c.get("iou")),
        "contour_overlap": _r(c.get("contour_overlap")),
        "chamfer_score": _r(c.get("chamfer_score")),
        "silhouette_score": _r(c.get("silhouette_score")),
        "edge_similarity": _r(c.get("edge_similarity")),
        "visual_similarity": _r(c.get("visual_similarity")),
        "final_score": _r(c.get("final_score")),
    }


def _save_debug_grid(ranked, output_dir: str):
    """Phase 9: montage of up to 8 Top-K renders labelled with their scores."""
    items = [c for c in ranked if c.get("_rgb") is not None][:8]
    if not items:
        return None
    cell = 220
    cols, rows = 4, 2
    canvas = np.full((rows * cell, cols * cell, 3), 30, np.uint8)
    for i, c in enumerate(items):
        r, col = divmod(i, cols)
        img = cv2.resize(c["_rgb"], (cell, cell - 40))
        y0, x0 = r * cell, col * cell
        canvas[y0:y0 + cell - 40, x0:x0 + cell] = img
        lines = [
            f"az{c['azimuth']:.0f} el{c['elevation']:.0f}",
            f"sil{c['silhouette_score']:.2f} vis{(c.get('visual_similarity') or 0):.2f}",
            f"final{c['final_score']:.2f}",
        ]
        for j, t in enumerate(lines):
            cv2.putText(canvas, t, (x0 + 4, y0 + cell - 28 + j * 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    path = os.path.join(output_dir, f"{uuid.uuid4()}_pose_grid.png")
    cv2.imwrite(path, canvas)
    return f"/outputs/{os.path.basename(path)}"


def _save_mask(mask: np.ndarray, output_dir: str, suffix: str) -> str:
    path = os.path.join(output_dir, f"{uuid.uuid4()}_{suffix}.png")
    cv2.imwrite(path, mask)
    return path
