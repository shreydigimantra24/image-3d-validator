"""
Texture Validator Service

Evaluates:
  1. Texture Presence (materials, UV coords, texture files)
  2. Perceptual Comparison (SSIM, LPIPS)

Texture Score = 0.2 * texture_presence + 0.4 * ssim_score + 0.4 * lpips_score
"""

import numpy as np
import trimesh
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def validate_texture(
    source_image_path: str,
    rendered_image_path: str,
    glb_path: str,
) -> dict:
    """
    Run full texture validation.

    Returns:
        dict with 'score' (0-100) and 'details' dict.
    """
    # Texture presence check
    presence = _texture_presence(glb_path)

    # Perceptual comparison
    perceptual = _perceptual_comparison(source_image_path, rendered_image_path)

    # Combined score
    score = round(
        0.2 * presence["score"]
        + 0.4 * perceptual["ssim_score"]
        + 0.4 * perceptual["lpips_score"],
        1,
    )
    score = max(0, min(100, score))

    return {
        "score": score,
        "details": {
            "texture_presence": presence,
            "perceptual": perceptual,
        },
    }


# ──────────────── Texture Presence ────────────────


def _texture_presence(glb_path: str) -> dict:
    """Check if the GLB has proper materials, textures, and UV mapping."""
    scene = trimesh.load(glb_path, force="scene")

    has_material = False
    has_texture = False
    has_uv = False
    material_count = 0
    texture_count = 0

    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue

        # Check material
        if geom.visual is not None:
            material = getattr(geom.visual, "material", None)
            if material is not None:
                has_material = True
                material_count += 1

                # Check texture image
                if hasattr(material, "image") and material.image is not None:
                    has_texture = True
                    texture_count += 1
                elif hasattr(material, "baseColorTexture") and material.baseColorTexture is not None:
                    has_texture = True
                    texture_count += 1

            # Check UV coordinates
            if hasattr(geom.visual, "uv") and geom.visual.uv is not None:
                if len(geom.visual.uv) > 0:
                    has_uv = True

    # Score components
    material_score = 100 if has_material else 0
    texture_score = 100 if has_texture else 0
    uv_score = 100 if has_uv else 0

    presence_score = 0.3 * material_score + 0.4 * texture_score + 0.3 * uv_score

    return {
        "score": round(presence_score, 1),
        "checks": {
            "has_material": has_material,
            "has_texture": has_texture,
            "has_uv_coordinates": has_uv,
            "material_count": material_count,
            "texture_count": texture_count,
        },
    }


# ──────────────── Perceptual Comparison ────────────────


def _load_and_prepare(image_path: str, target_size: tuple = (256, 256)) -> np.ndarray:
    """Load and resize an image to a common size for comparison."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert BGRA to BGR if needed
    if img.shape[-1] == 4:
        # Composite alpha over white background
        alpha = img[:, :, 3:4] / 255.0
        rgb = img[:, :, :3]
        white = np.ones_like(rgb) * 255
        img = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)

    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def _compute_ssim(source_path: str, rendered_path: str) -> float:
    """Compute Structural Similarity Index between two images."""
    img1 = _load_and_prepare(source_path)
    img2 = _load_and_prepare(rendered_path)

    # Convert to grayscale for SSIM
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    score, _ = ssim(gray1, gray2, full=True)
    return float(score)


def _compute_lpips(source_path: str, rendered_path: str) -> float:
    """
    Compute LPIPS perceptual distance.
    Falls back to a simpler perceptual metric if LPIPS is unavailable.
    """
    try:
        import torch
        import lpips

        loss_fn = lpips.LPIPS(net="alex")

        img1 = _load_and_prepare(source_path)
        img2 = _load_and_prepare(rendered_path)

        # Convert BGR to RGB and normalize to [-1, 1]
        t1 = torch.from_numpy(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 127.5 - 1.0
        t2 = torch.from_numpy(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 127.5 - 1.0

        t1 = t1.unsqueeze(0)
        t2 = t2.unsqueeze(0)

        with torch.no_grad():
            distance = loss_fn(t1, t2).item()

        return distance

    except ImportError:
        # Fallback: use normalized pixel-level difference as proxy
        img1 = _load_and_prepare(source_path).astype(np.float32) / 255.0
        img2 = _load_and_prepare(rendered_path).astype(np.float32) / 255.0
        mse = np.mean((img1 - img2) ** 2)
        return float(np.clip(mse * 2, 0, 1))  # Scale to roughly match LPIPS range


def _perceptual_comparison(source_path: str, rendered_path: str) -> dict:
    """Run SSIM and LPIPS comparisons."""
    ssim_val = _compute_ssim(source_path, rendered_path)
    lpips_val = _compute_lpips(source_path, rendered_path)

    # Convert to 0-100 scores
    ssim_score = ssim_val * 100  # SSIM is 0-1, higher is better
    lpips_score = max(0, (1 - lpips_val) * 100)  # LPIPS is 0-1, lower is better

    return {
        "ssim_raw": round(ssim_val, 4),
        "lpips_raw": round(lpips_val, 4),
        "ssim_score": round(ssim_score, 1),
        "lpips_score": round(lpips_score, 1),
    }
