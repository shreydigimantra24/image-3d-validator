"""
Geometry Quality Checker

Detects structural mesh defects that a silhouette match alone cannot catch.

ASSET-CLASS AWARE (Fix 1)
-------------------------
Product / assembly meshes (furniture, appliances) are composed of MANY separate
open-shell parts BY DESIGN: a table + 4 chairs is thousands of disconnected
components (tabletop, seat pans, legs, plus hundreds of tiny screws/glides), it
is NOT watertight, and it has tens of thousands of open boundary edges. None of
those are defects for this asset class, so they must NOT cap the geometry score.

We therefore split the checks into two buckets:

  * Topology descriptors (informational): watertight, component count, open
    boundary "holes", non-manifold edges. Reported, but only gate a
    `single_solid` asset (e.g. a scan / printable part).

  * Genuine defects (always meaningful): NaN/inf vertices, inconsistent normals,
    zero-area (degenerate) faces, isolated 1-2 face slivers floating away from
    the body, and substantial components flung far outside the main bounding box
    (true floaters / spikes).

The geometry SCORE itself is computed render-first in geometry_validator.py;
this module supplies the defect signals and an asset-aware gate that only fires
on genuine defects.
"""

import numpy as np
import trimesh

from services.mesh_cache import load_combined
from services.validation_config import (
    DEFAULT_ASSET_CLASS,
    ASSEMBLY_MIN_COMPONENTS,
    ASSEMBLY_MIN_SUBSTANTIAL_PARTS,
    SUBSTANTIAL_PART_MIN_FACES,
)


def _component_faces(mesh: trimesh.Trimesh):
    """Return a list of face-index arrays, one per connected component.

    Uses face-adjacency graph traversal (no `mesh.split()`, which instantiates a
    full Trimesh per component and blows up RAM on noisy/high-poly meshes).
    """
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return []
    return trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(n_faces)
    )


def _count_components(mesh: trimesh.Trimesh) -> int:
    comps = _component_faces(mesh)
    return len(comps)


def check_geometry_quality(glb_path: str) -> dict:
    """
    Analyze mesh structural quality.

    Returns dict with topology descriptors AND genuine-defect signals:
        {
          # topology descriptors (informational; only gate single_solid)
          "holes": int, "non_manifold_edges": int, "components": int,
          "is_watertight": bool,
          # genuine defects (always meaningful)
          "degenerate_faces": int, "normals_consistent": bool,
          "nan_or_inf_vertices": bool, "isolated_slivers": int, "far_floaters": int,
          # part structure (drives assembly auto-detection)
          "substantial_components": int,
          "total_faces": int, "total_vertices": int
        }
    """
    combined = load_combined(glb_path)
    if combined is None:
        return {
            "holes": -1,
            "non_manifold_edges": -1,
            "degenerate_faces": -1,
            "components": 0,
            "substantial_components": 0,
            "normals_consistent": False,
            "nan_or_inf_vertices": True,
            "isolated_slivers": 0,
            "far_floaters": 0,
            "total_faces": 0,
            "total_vertices": 0,
            "is_watertight": False,
            "error": "No valid meshes found",
        }

    comps = _component_faces(combined)
    sizes = np.array([len(c) for c in comps]) if comps else np.array([], dtype=int)
    substantial = int(np.sum(sizes >= SUBSTANTIAL_PART_MIN_FACES))
    slivers, floaters = _isolated_slivers_and_floaters(combined, comps, sizes)

    return {
        "holes": _count_holes(combined),
        "non_manifold_edges": _count_non_manifold_edges(combined),
        "degenerate_faces": _count_degenerate_faces(combined),
        "components": len(comps),
        "substantial_components": substantial,
        "normals_consistent": _normals_consistent(combined),
        "nan_or_inf_vertices": _has_nan_or_inf(combined),
        "isolated_slivers": slivers,
        "far_floaters": floaters,
        "total_faces": int(len(combined.faces)),
        "total_vertices": int(len(combined.vertices)),
        "is_watertight": bool(combined.is_watertight),
    }


def classify_asset(quality: dict, override: str = None) -> str:
    """
    Decide whether the mesh is a multi-part product/assembly or a single solid.

    An explicit `override` (from the API request / config) always wins. Otherwise
    we auto-detect: a mesh with many connected components AND several substantial
    parts is an assembly (table + chairs, screws and all).
    """
    if override in ("product", "single_solid"):
        return override
    components = quality.get("components", 1) or 1
    substantial = quality.get("substantial_components", 0) or 0
    if (
        components >= ASSEMBLY_MIN_COMPONENTS
        and substantial >= ASSEMBLY_MIN_SUBSTANTIAL_PARTS
    ):
        return "product"
    return DEFAULT_ASSET_CLASS


def apply_geometry_gates(base_score: float, quality: dict, asset_class: str = None) -> dict:
    """
    Cap the geometry score ONLY for genuine, asset-appropriate defects.

    For `product`/assembly meshes we never cap for non-watertightness, component
    count, or open boundary edges — those are by-design for multi-part assets.
    We still cap for defects that break any mesh: NaN/inf geometry, inconsistent
    normals, a high ratio of zero-area faces, and true floaters/spikes.

    For `single_solid` meshes the legacy topology gates still apply.

    Returns dict: { "score": float, "gated": bool, "applied_gates": [str, ...],
                    "asset_class": str }
    """
    asset_class = classify_asset(quality, asset_class)
    score = float(base_score)
    applied = []

    degenerate = quality.get("degenerate_faces", 0) or 0
    non_manifold = quality.get("non_manifold_edges", 0) or 0
    total_faces = max(quality.get("total_faces", 1), 1)
    far_floaters = quality.get("far_floaters", 0) or 0

    # ── Genuine defects — gate every asset class ──
    if quality.get("nan_or_inf_vertices"):
        score = min(score, 20)
        applied.append("NaN/inf vertices → cap 20")

    if not quality.get("normals_consistent", True):
        score = min(score, 70)
        applied.append("inconsistent normals → cap 70")

    if degenerate > 0 and degenerate / total_faces > 0.05:
        score = min(score, 55)
        applied.append(f"degenerate faces {degenerate}/{total_faces} → cap 55")

    if far_floaters > 0:
        score = min(score, 70)
        applied.append(f"{far_floaters} detached floater part(s) far from body → cap 70")

    # ── Topology gates — single_solid ONLY (meaningless for assemblies) ──
    if asset_class == "single_solid":
        holes = quality.get("holes", 0)
        components = quality.get("components", 1)
        if holes is not None and holes > 10:
            score = min(score, 40)
            applied.append(f"holes>10 ({holes}) → cap 40")
        if components > 5:
            score = min(score, 50)
            applied.append(f"floating components ({components}) → cap 50")
        if non_manifold is not None and non_manifold > 50:
            score = min(score, 60)
            applied.append(f"non-manifold edges ({non_manifold}) → cap 60")

    return {
        "score": round(max(0.0, score), 1),
        "gated": len(applied) > 0,
        "applied_gates": applied,
        "asset_class": asset_class,
    }


# ──────────────── Individual checks ────────────────


def _count_holes(mesh: trimesh.Trimesh) -> int:
    """Number of open boundary loops (proxy for holes / open shells).

    NOTE: for assembly meshes this is large by design (every open-shell part
    contributes a boundary) and is informational only — it does NOT cap the
    product geometry score.
    """
    try:
        # Edges referenced by exactly one face are boundary edges.
        groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
        boundary_edges = len(groups)
        if boundary_edges == 0:
            return 0
        # Estimate loops via connected boundary outline; fall back to edge count.
        try:
            outline = mesh.outline()
            entities = getattr(outline, "entities", None)
            if entities is not None and len(entities) > 0:
                return int(len(entities))
        except Exception:
            pass
        return int(boundary_edges)
    except Exception:
        return 0


def _count_non_manifold_edges(mesh: trimesh.Trimesh) -> int:
    """Edges shared by a number of faces other than 2."""
    try:
        edges_sorted = mesh.edges_sorted
        # Count how many faces reference each unique edge.
        unique, counts = np.unique(edges_sorted, axis=0, return_counts=True)
        # Manifold interior edge -> 2 faces; boundary edge -> 1. Anything else
        # (>2) is non-manifold. Count >2 occurrences.
        return int(np.sum(counts > 2))
    except Exception:
        return 0


def _count_degenerate_faces(mesh: trimesh.Trimesh, eps: float = 1e-10) -> int:
    try:
        return int(np.sum(mesh.area_faces < eps))
    except Exception:
        return 0


def _normals_consistent(mesh: trimesh.Trimesh) -> bool:
    """True if face winding is coherent (consistent outward orientation)."""
    try:
        if hasattr(mesh, "is_winding_consistent"):
            return bool(mesh.is_winding_consistent)
    except Exception:
        pass
    return True


def _has_nan_or_inf(mesh: trimesh.Trimesh) -> bool:
    """True if any vertex coordinate is NaN/inf — a hard, render-breaking defect."""
    try:
        return bool(not np.all(np.isfinite(mesh.vertices)))
    except Exception:
        return False


def _isolated_slivers_and_floaters(mesh, comps, sizes):
    """
    Count two kinds of GENUINE geometry artifacts (asset-class independent):

      * isolated slivers — components of 1-2 faces whose centroid sits well
        outside the main body (stray triangles, not part of any real surface).
      * far floaters      — substantial components (>= SUBSTANTIAL_PART_MIN_FACES)
        whose centroid is flung far outside the main body's bounding box (a part
        modeled in the wrong place / a spike).

    Tiny fittings (screws, glides) that sit ON the furniture are intentionally
    NOT counted: they are close to the body, so they fail the distance test.
    """
    if not comps or len(comps) <= 1:
        return 0, 0
    try:
        face_centers = mesh.triangles_center  # (n_faces, 3)
        # Main body = largest component; measure distances against its extent.
        main_idx = int(np.argmax(sizes))
        main_faces = comps[main_idx]
        main_pts = face_centers[main_faces]
        main_center = main_pts.mean(axis=0)
        main_diag = float(np.linalg.norm(main_pts.max(axis=0) - main_pts.min(axis=0)))
        if main_diag < 1e-9:
            return 0, 0

        slivers = 0
        floaters = 0
        for i, faces in enumerate(comps):
            if i == main_idx:
                continue
            centroid = face_centers[faces].mean(axis=0)
            dist = float(np.linalg.norm(centroid - main_center)) / main_diag
            n = len(faces)
            if n <= 2 and dist > 0.6:
                slivers += 1
            elif n >= SUBSTANTIAL_PART_MIN_FACES and dist > 1.0:
                floaters += 1
        return int(slivers), int(floaters)
    except Exception:
        return 0, 0
