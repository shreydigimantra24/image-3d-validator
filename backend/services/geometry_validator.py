"""
Geometry Validator Service

Evaluates:
  1. Silhouette Matching (IoU, Chamfer Distance, Hausdorff Distance) — render-derived
  2. Structural soundness (genuine defects only, asset-class aware)

ASSET-CLASS AWARE SCORING (Fix 1)
---------------------------------
  product / assembly  →  score = structural_soundness × silhouette_factor(IoU)
        Render-first: a structurally sound multi-part asset (furniture) stays
        HIGH even though it is non-watertight with thousands of open-shell parts.
        It is NEVER penalised for non-watertightness, component count, or open
        boundary edges. The silhouette factor only pulls the score down when the
        rendered SHAPE genuinely disagrees with the photo.

  single_solid        →  score = 0.4 * mesh_integrity + 0.6 * silhouette  (legacy)
"""

import numpy as np
import trimesh
import cv2
from scipy.spatial.distance import directed_hausdorff

from services.mesh_cache import load_combined
from services.geometry_quality_checker import check_geometry_quality, classify_asset


def _count_components(mesh: trimesh.Trimesh) -> int:
    """
    Count connected components via lightweight graph traversal over face
    adjacency. Avoids `mesh.split()`, which instantiates a full Trimesh per
    component and blows up RAM on noisy/high-poly meshes.
    """
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return 0
    components = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(n_faces)
    )
    return len(components)


def validate_geometry(
    source_image_path: str,
    glb_path: str,
    rendered_image_path: str,
    asset_class: str = None,
    quality: dict = None,
) -> dict:
    """
    Run full geometry validation.

    Args:
        asset_class: "product" | "single_solid" | None (auto-detect).
        quality: precomputed check_geometry_quality(glb_path) dict; recomputed
            here if not supplied (keeps the function standalone-callable).

    Returns:
        dict with 'score' (0-100) and 'details' dict.
    """
    if quality is None:
        quality = check_geometry_quality(glb_path)
    asset_class = classify_asset(quality, asset_class)

    # Silhouette matching (render-derived shape agreement).
    silhouette = _silhouette_matching(source_image_path, rendered_image_path)
    iou = silhouette["metrics"]["iou"]

    if asset_class == "product":
        # Render-first: structurally sound assembly stays HIGH; the shape factor
        # only reduces it when the rendered outline genuinely disagrees. No
        # penalty for watertightness / component count / open boundary edges.
        structural = _structural_soundness(quality)
        factor = _silhouette_factor(iou)
        score = round(structural * factor, 1)
        integrity = {
            "score": round(structural, 1),
            "mode": "product_structural",
            "silhouette_factor": round(factor, 3),
            "checks": _integrity_checks(quality),
        }
    else:
        integrity = _mesh_integrity_legacy(quality)
        score = round(0.4 * integrity["score"] + 0.6 * silhouette["score"], 1)

    score = max(0, min(100, score))

    return {
        "score": score,
        "details": {
            "asset_class": asset_class,
            "mesh_integrity": integrity,
            "silhouette_matching": silhouette,
        },
    }


# ──────────────── Structural soundness (product) ────────────────


def _structural_soundness(quality: dict) -> float:
    """
    0-100 soundness from GENUINE defects only (asset-class independent signals).
    Starts at 100 and subtracts for real problems — never for non-watertightness,
    component count, or open boundary edges (by-design for assemblies).
    """
    score = 100.0
    total_faces = max(quality.get("total_faces", 1), 1)

    if quality.get("nan_or_inf_vertices"):
        return 0.0  # render-breaking; nothing else matters

    if not quality.get("normals_consistent", True):
        score -= 20.0

    degenerate = quality.get("degenerate_faces", 0) or 0
    degen_ratio = degenerate / total_faces
    if degen_ratio > 0.001:
        score -= min(30.0, degen_ratio * 200.0)

    far_floaters = quality.get("far_floaters", 0) or 0
    if far_floaters > 0:
        score -= min(30.0, far_floaters * 10.0)

    slivers = quality.get("isolated_slivers", 0) or 0
    if slivers > 0:
        score -= min(15.0, slivers * 3.0)

    return float(max(0.0, min(100.0, score)))


def _silhouette_factor(iou: float) -> float:
    """
    Map silhouette IoU to a multiplicative shape-agreement factor in [0.6, 1.0].
    A structurally perfect mesh with a moderate (but real) alignment keeps most
    of its score; only a catastrophic shape mismatch drives it down hard.
    """
    try:
        v = float(iou)
    except (TypeError, ValueError):
        return 0.85
    if v >= 0.85:
        return 1.0
    if v >= 0.5:
        return 0.9 + (v - 0.5) / 0.35 * 0.1
    if v >= 0.3:
        return 0.8 + (v - 0.3) / 0.2 * 0.1
    return 0.6 + max(0.0, v) / 0.3 * 0.2


def _integrity_checks(quality: dict) -> dict:
    """Surface the descriptors + genuine-defect signals (no derived score caps)."""
    return {
        "is_watertight": quality.get("is_watertight"),
        "num_components": quality.get("components"),
        "substantial_components": quality.get("substantial_components"),
        "degenerate_faces": quality.get("degenerate_faces"),
        "normals_consistent": quality.get("normals_consistent"),
        "nan_or_inf_vertices": quality.get("nan_or_inf_vertices"),
        "isolated_slivers": quality.get("isolated_slivers"),
        "far_floaters": quality.get("far_floaters"),
        "total_faces": quality.get("total_faces"),
        "vertex_count": quality.get("total_vertices"),
    }


def _mesh_integrity_legacy(quality: dict) -> dict:
    """Legacy topology-weighted integrity for single_solid assets, derived from
    the shared quality dict (no extra mesh load)."""
    is_watertight = bool(quality.get("is_watertight"))
    num_components = quality.get("components", 1) or 1
    has_floating = num_components > 1
    normals_ok = quality.get("normals_consistent", True)
    degenerate_count = quality.get("degenerate_faces", 0) or 0
    total_faces = max(quality.get("total_faces", 1), 1)
    degenerate_ratio = degenerate_count / total_faces

    watertight_score = 100 if is_watertight else 40
    manifold_score = 100 if normals_ok else 50
    component_score = 100 if not has_floating else max(0, 100 - (num_components - 1) * 20)
    degenerate_score = max(0, 100 - degenerate_ratio * 500)

    integrity_score = (
        0.30 * watertight_score
        + 0.25 * manifold_score
        + 0.25 * component_score
        + 0.20 * degenerate_score
    )

    return {
        "score": round(integrity_score, 1),
        "mode": "single_solid_legacy",
        "checks": {
            "is_watertight": is_watertight,
            "num_components": num_components,
            "has_floating_components": has_floating,
            "degenerate_faces": degenerate_count,
            "total_faces": total_faces,
            "degenerate_ratio": round(degenerate_ratio, 4),
            "vertex_count": quality.get("total_vertices"),
        },
    }


# ──────────────── Silhouette Matching ────────────────


def _extract_mask(image_path: str) -> np.ndarray:
    """Extract a binary foreground mask from an image."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # If image has alpha channel, use that as mask
    if img.shape[-1] == 4:
        mask = (img[:, :, 3] > 128).astype(np.uint8) * 255
    else:
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    return mask


def _silhouette_matching(source_path: str, rendered_path: str) -> dict:
    """Compare silhouettes of source and rendered images."""
    mask_src = _extract_mask(source_path)
    mask_rnd = _extract_mask(rendered_path)

    # Resize to common resolution
    target_size = (256, 256)
    mask_src = cv2.resize(mask_src, target_size, interpolation=cv2.INTER_NEAREST)
    mask_rnd = cv2.resize(mask_rnd, target_size, interpolation=cv2.INTER_NEAREST)

    # IoU
    intersection = np.logical_and(mask_src > 0, mask_rnd > 0).sum()
    union = np.logical_or(mask_src > 0, mask_rnd > 0).sum()
    iou = intersection / max(union, 1)

    # Contour extraction for distance metrics
    contours_src, _ = cv2.findContours(mask_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_rnd, _ = cv2.findContours(mask_rnd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours_src and contours_rnd:
        pts_src = np.vstack(contours_src).squeeze()
        pts_rnd = np.vstack(contours_rnd).squeeze()

        # Ensure 2D
        if pts_src.ndim == 1:
            pts_src = pts_src.reshape(1, -1)
        if pts_rnd.ndim == 1:
            pts_rnd = pts_rnd.reshape(1, -1)

        # Hausdorff Distance (normalized by diagonal)
        diag = np.sqrt(target_size[0] ** 2 + target_size[1] ** 2)
        hausdorff = max(
            directed_hausdorff(pts_src, pts_rnd)[0],
            directed_hausdorff(pts_rnd, pts_src)[0],
        )
        hausdorff_norm = hausdorff / diag

        # Chamfer Distance
        from scipy.spatial import cKDTree

        tree_src = cKDTree(pts_src)
        tree_rnd = cKDTree(pts_rnd)
        d_src, _ = tree_rnd.query(pts_src)
        d_rnd, _ = tree_src.query(pts_rnd)
        chamfer = (d_src.mean() + d_rnd.mean()) / 2 / diag
    else:
        hausdorff_norm = 1.0
        chamfer = 1.0

    # Score: combine IoU and distance metrics
    iou_score = iou * 100
    hausdorff_score = max(0, 100 - hausdorff_norm * 300)
    chamfer_score = max(0, 100 - chamfer * 500)

    silhouette_score = 0.5 * iou_score + 0.25 * hausdorff_score + 0.25 * chamfer_score

    return {
        "score": round(silhouette_score, 1),
        "metrics": {
            "iou": round(iou, 4),
            "hausdorff_distance_normalized": round(hausdorff_norm, 4),
            "chamfer_distance_normalized": round(chamfer, 4),
        },
    }
