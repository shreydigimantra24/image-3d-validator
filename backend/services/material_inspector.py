"""
Material Inspector (Fix 3)

Reads PBR material properties straight from the GLB and exposes:

  * metallic / roughness / baseColor factors
  * the baseColor TEXTURE (albedo) pixels — the asset's true surface color,
    independent of how the renderer lit it
  * MATERIAL warnings (separate from the color score)

WHY THIS EXISTS
---------------
The reference asset ships with metallicFactor = 1.0. A fully metallic surface
rendered WITHOUT an environment map / IBL reflects "nothing" and goes dark, so
the render shows charcoal chairs even though the albedo texture is light
(mean RGB ~[204,205,206]). Blaming the asset's COLOR for the renderer's lighting
is wrong. We therefore (a) flag suspicious metallic as a MATERIAL warning and
(b) hand the albedo texture to the color stage so color is judged on the asset's
real surface color, not an unlit metallic render.
"""

import numpy as np
import trimesh

from services.mesh_cache import load_scene
from services.validation_config import METALLIC_WARN_THRESHOLD

ALBEDO_MAX_SIZE = 256  # downscale the (often 4K) baseColor texture before use


def inspect_material(glb_path: str) -> dict:
    """
    Inspect PBR material properties of the GLB.

    Returns dict:
        {
          "metallic_factor": float|None, "roughness_factor": float|None,
          "base_color_factor": [r,g,b,a]|None,   # 0-1
          "has_base_color_texture": bool,
          "suspicious_metallic": bool,
          "warnings": [str, ...]
        }
    """
    scene = load_scene(glb_path)
    metallic = None
    roughness = None
    base_color_factor = None
    has_texture = False

    for geom in scene.geometry.values():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        mat = getattr(getattr(geom, "visual", None), "material", None)
        if mat is None:
            continue
        m = _get_float(mat, "metallicFactor")
        r = _get_float(mat, "roughnessFactor")
        bcf = getattr(mat, "baseColorFactor", None)
        if m is not None:
            metallic = m if metallic is None else max(metallic, m)
        if r is not None:
            roughness = r if roughness is None else max(roughness, r)
        if bcf is not None and base_color_factor is None:
            base_color_factor = [float(v) for v in np.array(bcf).ravel()[:4]]
        if _material_albedo_image(mat) is not None:
            has_texture = True

    warnings = []
    suspicious = False
    if (
        metallic is not None
        and metallic > METALLIC_WARN_THRESHOLD
        and has_texture
    ):
        suspicious = True
        warnings.append(
            f"metallicFactor={round(metallic, 2)} on a textured asset — likely an "
            f"export bug; a fully-metallic surface renders dark without an "
            f"environment map even though the albedo texture is correct."
        )

    return {
        "metallic_factor": metallic,
        "roughness_factor": roughness,
        "base_color_factor": base_color_factor,
        "has_base_color_texture": has_texture,
        "suspicious_metallic": suspicious,
        "warnings": warnings,
    }


def get_albedo_rgb(glb_path: str) -> "np.ndarray|None":
    """
    Return the asset's baseColor (albedo) as an HxWx3 uint8 RGB array, modulated
    by baseColorFactor, downscaled to ALBEDO_MAX_SIZE. None if the asset has no
    baseColor texture (caller then falls back to the exposure-normalized render).
    """
    scene = load_scene(glb_path)
    for geom in scene.geometry.values():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        mat = getattr(getattr(geom, "visual", None), "material", None)
        if mat is None:
            continue
        img = _material_albedo_image(mat)
        if img is None:
            continue
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        # Downscale large textures (4K → 256) — palette/mean are unaffected.
        h, w = rgb.shape[:2]
        scale = ALBEDO_MAX_SIZE / max(h, w)
        if scale < 1.0:
            import cv2

            rgb = cv2.resize(
                rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        # Modulate by baseColorFactor RGB if present (glTF multiplies them).
        bcf = getattr(mat, "baseColorFactor", None)
        if bcf is not None:
            f = np.clip(np.array(bcf, dtype=np.float32).ravel()[:3], 0.0, 1.0)
            rgb = np.clip(rgb.astype(np.float32) * f, 0, 255).astype(np.uint8)
        return rgb
    return None


# ──────────────── Helpers ────────────────


def _get_float(mat, attr):
    val = getattr(mat, attr, None)
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _material_albedo_image(mat):
    """Return the PIL baseColor/diffuse image for a trimesh material, or None."""
    for attr in ("baseColorTexture", "image"):
        img = getattr(mat, attr, None)
        if img is not None and hasattr(img, "convert"):
            return img
    return None
