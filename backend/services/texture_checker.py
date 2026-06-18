"""
Texture Presence Checker (Enhancement 4)

Verifies texture completeness BEFORE perceptual texture similarity is trusted.
A model with no material / no UVs / no texture image must not earn a moderate
texture score from coincidental pixel similarity.

Checks:
  * Material exists      — mesh.visual.material
  * UV coordinates exist — mesh.visual.uv
  * Texture image exists — material.image / baseColorTexture
"""

import trimesh

from services.mesh_cache import load_scene


def check_texture_presence(glb_path: str) -> dict:
    """
    Inspect the GLB for material, UV, and texture-image presence.

    Returns dict:
        {
          "material_present": bool, "uv_present": bool, "texture_present": bool,
          "material_count": int, "texture_count": int,
          "has_vertex_colors": bool
        }
    """
    scene = load_scene(glb_path)

    material_present = False
    uv_present = False
    texture_present = False
    has_vertex_colors = False
    material_count = 0
    texture_count = 0
    texture_resolution = None  # (width, height) of the baseColor texture

    for geom in scene.geometry.values():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        visual = getattr(geom, "visual", None)
        if visual is None:
            continue

        material = getattr(visual, "material", None)
        if material is not None:
            material_present = True
            material_count += 1
            if _material_has_image(material):
                texture_present = True
                texture_count += 1
                if texture_resolution is None:
                    texture_resolution = _material_texture_size(material)

        uv = getattr(visual, "uv", None)
        if uv is not None and len(uv) > 0:
            uv_present = True

        # Vertex colors are a partial substitute for textures.
        if getattr(visual, "kind", None) == "vertex":
            vc = getattr(visual, "vertex_colors", None)
            if vc is not None and len(vc) > 0:
                has_vertex_colors = True

    return {
        "material_present": material_present,
        "uv_present": uv_present,
        "texture_present": texture_present,
        "material_count": material_count,
        "texture_count": texture_count,
        "has_vertex_colors": has_vertex_colors,
        "texture_resolution": list(texture_resolution) if texture_resolution else None,
    }


def texture_presence_score(presence: dict) -> float:
    """Weighted 0-100 presence score (material 0.3 / texture 0.4 / uv 0.3)."""
    material = 100 if presence.get("material_present") else 0
    texture = 100 if presence.get("texture_present") else 0
    uv = 100 if presence.get("uv_present") else 0
    return round(0.3 * material + 0.4 * texture + 0.3 * uv, 1)


def apply_texture_gates(base_score: float, presence: dict) -> dict:
    """
    Penalize the texture score when required texture data is missing.

    Rules (from the plan):
      * Missing material     → 0-10
      * Missing texture file → heavy penalty
      * Missing UVs          → heavy penalty

    Returns dict: { "score": float, "gated": bool, "applied_gates": [str,...] }
    """
    score = float(base_score)
    applied = []

    if not presence.get("material_present"):
        score = min(score, 10)
        applied.append("no material → cap 10")

    if not presence.get("texture_present"):
        if presence.get("has_vertex_colors"):
            score = min(score, 50)
            applied.append("no texture image (vertex colors only) → cap 50")
        else:
            score = min(score, 25)
            applied.append("no texture image → cap 25")

    if not presence.get("uv_present") and not presence.get("has_vertex_colors"):
        score = min(score, 30)
        applied.append("no UV coordinates → cap 30")

    return {
        "score": round(max(0.0, score), 1),
        "gated": len(applied) > 0,
        "applied_gates": applied,
    }


# ──────────────── Helpers ────────────────


def _material_has_image(material) -> bool:
    if hasattr(material, "image") and getattr(material, "image") is not None:
        return True
    if hasattr(material, "baseColorTexture") and getattr(material, "baseColorTexture") is not None:
        return True
    return False


def _material_texture_size(material):
    """Return (width, height) of the material's baseColor/diffuse image, or None."""
    for attr in ("baseColorTexture", "image"):
        img = getattr(material, attr, None)
        size = getattr(img, "size", None)  # PIL images expose .size = (w, h)
        if size and len(size) == 2:
            return (int(size[0]), int(size[1]))
    return None
