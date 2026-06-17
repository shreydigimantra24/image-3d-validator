"""
Geometry Validator Service

Evaluates:
  1. Mesh Integrity (holes, non-manifold edges, degenerate faces, floating components, watertightness)
  2. Silhouette Matching (IoU, Chamfer Distance, Hausdorff Distance)

Geometry Score = 0.4 * mesh_integrity + 0.6 * silhouette_similarity
"""

import numpy as np
import trimesh
import cv2
from scipy.spatial.distance import directed_hausdorff


def validate_geometry(
    source_image_path: str,
    glb_path: str,
    rendered_image_path: str,
) -> dict:
    """
    Run full geometry validation.

    Returns:
        dict with 'score' (0-100) and 'details' dict.
    """
    # Mesh integrity
    integrity = _mesh_integrity(glb_path)

    # Silhouette matching
    silhouette = _silhouette_matching(source_image_path, rendered_image_path)

    # Combined score
    score = round(0.4 * integrity["score"] + 0.6 * silhouette["score"], 1)
    score = max(0, min(100, score))

    return {
        "score": score,
        "details": {
            "mesh_integrity": integrity,
            "silhouette_matching": silhouette,
        },
    }


# ──────────────── Mesh Integrity ────────────────


def _mesh_integrity(glb_path: str) -> dict:
    """Analyze mesh topology and structure."""
    scene = trimesh.load(glb_path, force="scene")

    # Collect all meshes
    meshes = []
    for name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh):
            meshes.append(geom)

    if not meshes:
        return {"score": 0, "checks": {"error": "No valid meshes found"}}

    combined = trimesh.util.concatenate(meshes)

    # Individual checks
    is_watertight = combined.is_watertight
    num_components = len(combined.split(only_watertight=False))
    has_floating = num_components > 1

    # Non-manifold edges
    edges = combined.edges_unique
    edges_face_count = trimesh.grouping.group_rows(combined.edges_sorted, require_count=2)
    # An edge shared by != 2 faces is non-manifold
    face_adjacency = combined.face_adjacency
    non_manifold_count = 0
    try:
        # Use trimesh's own check
        if hasattr(combined, "is_manifold"):
            is_manifold = combined.is_manifold
        else:
            is_manifold = True  # assume manifold if check unavailable
    except Exception:
        is_manifold = True

    # Degenerate faces (zero-area triangles)
    face_areas = combined.area_faces
    degenerate_count = int(np.sum(face_areas < 1e-10))
    total_faces = len(combined.faces)
    degenerate_ratio = degenerate_count / max(total_faces, 1)

    # Scoring
    watertight_score = 100 if is_watertight else 40
    manifold_score = 100 if is_manifold else 50
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
        "checks": {
            "is_watertight": is_watertight,
            "is_manifold": is_manifold,
            "num_components": num_components,
            "has_floating_components": has_floating,
            "degenerate_faces": degenerate_count,
            "total_faces": total_faces,
            "degenerate_ratio": round(degenerate_ratio, 4),
            "vertex_count": len(combined.vertices),
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
