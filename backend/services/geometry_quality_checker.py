"""
Geometry Quality Checker (Enhancement 3)

Detects structural mesh defects that a silhouette match alone cannot catch:

  * Holes               — open boundary loops / missing regions
  * Non-manifold edges  — edges shared by != 2 faces
  * Degenerate faces    — zero-area triangles
  * Floating components — disconnected mesh fragments
  * Normal consistency  — coherent face winding/orientation

These feed a score-gating layer so a geometrically broken mesh cannot keep a
high geometry score just because its silhouette lines up.
"""

import numpy as np
import trimesh

from services.mesh_cache import load_combined


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


def check_geometry_quality(glb_path: str) -> dict:
    """
    Analyze mesh structural quality.

    Returns dict:
        {
          "holes": int, "non_manifold_edges": int, "degenerate_faces": int,
          "components": int, "normals_consistent": bool,
          "total_faces": int, "total_vertices": int,
          "is_watertight": bool
        }
    """
    combined = load_combined(glb_path)
    if combined is None:
        return {
            "holes": -1,
            "non_manifold_edges": -1,
            "degenerate_faces": -1,
            "components": 0,
            "normals_consistent": False,
            "total_faces": 0,
            "total_vertices": 0,
            "is_watertight": False,
            "error": "No valid meshes found",
        }

    return {
        "holes": _count_holes(combined),
        "non_manifold_edges": _count_non_manifold_edges(combined),
        "degenerate_faces": _count_degenerate_faces(combined),
        "components": _count_components(combined),
        "normals_consistent": _normals_consistent(combined),
        "total_faces": int(len(combined.faces)),
        "total_vertices": int(len(combined.vertices)),
        "is_watertight": bool(combined.is_watertight),
    }


def apply_geometry_gates(base_score: float, quality: dict) -> dict:
    """
    Cap the geometry score when severe structural defects are present.

    Returns dict:
        { "score": float, "gated": bool, "applied_gates": [str, ...] }
    """
    score = float(base_score)
    applied = []

    holes = quality.get("holes", 0)
    components = quality.get("components", 1)
    degenerate = quality.get("degenerate_faces", 0)
    non_manifold = quality.get("non_manifold_edges", 0)
    total_faces = max(quality.get("total_faces", 1), 1)

    if holes is not None and holes > 10:
        score = min(score, 40)
        applied.append(f"holes>{10} ({holes}) → cap 40")

    if components > 5:
        score = min(score, 50)
        applied.append(f"floating components ({components}) → cap 50")

    if degenerate > 0 and degenerate / total_faces > 0.05:
        score = min(score, 55)
        applied.append(f"degenerate faces {degenerate}/{total_faces} → cap 55")

    if non_manifold is not None and non_manifold > 50:
        score = min(score, 60)
        applied.append(f"non-manifold edges ({non_manifold}) → cap 60")

    if not quality.get("normals_consistent", True):
        score = min(score, 70)
        applied.append("inconsistent normals → cap 70")

    return {
        "score": round(max(0.0, score), 1),
        "gated": len(applied) > 0,
        "applied_gates": applied,
    }


# ──────────────── Individual checks ────────────────


def _count_holes(mesh: trimesh.Trimesh) -> int:
    """Number of open boundary loops (proxy for holes)."""
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
        # trimesh exposes is_winding_consistent on triangulated meshes.
        if hasattr(mesh, "is_winding_consistent"):
            return bool(mesh.is_winding_consistent)
    except Exception:
        pass
    return True
