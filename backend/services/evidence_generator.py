"""
Validation Evidence Generator (Enhancement 2)

Produces a blended overlay of the input image and the aligned render so a
reviewer can visually confirm what the validator compared and that the camera
pose lines up. The overlay is surfaced in the UI "Validation Evidence" panel.
"""

import os
import uuid
import numpy as np
import cv2


def generate_overlay(source_image_path: str, aligned_render_path: str, output_dir: str) -> dict:
    """
    Blend source image and aligned render 50/50.

    Returns dict:
        { "overlay_path": str, "overlay_url": str }
    """
    src = _load_rgb(source_image_path)
    rnd = _load_rgb(aligned_render_path)

    h, w = src.shape[:2]
    rnd = cv2.resize(rnd, (w, h), interpolation=cv2.INTER_AREA)

    overlay = cv2.addWeighted(src, 0.5, rnd, 0.5, 0.0)

    overlay_path = os.path.join(output_dir, f"{uuid.uuid4()}_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return {
        "overlay_path": overlay_path,
        "overlay_url": f"/outputs/{os.path.basename(overlay_path)}",
    }


def _load_rgb(image_path: str) -> np.ndarray:
    """Load an image as RGB, compositing any alpha over white."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[-1] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        white = np.ones_like(rgb) * 255.0
        img = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
