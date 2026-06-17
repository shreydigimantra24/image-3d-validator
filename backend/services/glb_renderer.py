"""
GLB Renderer Service — renders a GLB model to a 2D image using Trimesh + PyRender.

Supports rendering from arbitrary camera poses (azimuth/elevation) so the
pose estimator can search for the viewpoint that best matches the input image.
"""

import os
import uuid
import numpy as np
import trimesh
from PIL import Image


# ──────────────── Public API ────────────────


def render_glb(glb_path: str, output_dir: str, resolution: tuple = (512, 512)) -> str:
    """
    Load a GLB file and render it to a 2D PNG image from the default
    front (+Z) viewpoint.

    Returns:
        Path to the rendered PNG image.
    """
    return render_glb_from_pose(
        glb_path, azimuth=0.0, elevation=0.0, output_dir=output_dir, resolution=resolution
    )


def render_glb_from_pose(
    glb_path: str,
    azimuth: float,
    elevation: float,
    output_dir: str,
    resolution: tuple = (512, 512),
    suffix: str = "rendered",
) -> str:
    """
    Render a GLB from a specific camera pose.

    Args:
        glb_path: Path to the GLB file.
        azimuth: Horizontal orbit angle in degrees (rotation about +Y).
        elevation: Vertical angle in degrees (positive looks down from above).
        output_dir: Directory to save the rendered image.
        resolution: Output image resolution (width, height).
        suffix: Filename suffix (before .png).

    Returns:
        Path to the rendered PNG image.
    """
    scene = trimesh.load(glb_path, force="scene")
    camera_pose = _camera_pose_for(scene, azimuth, elevation)

    try:
        return _render_with_pyrender(scene, output_dir, resolution, camera_pose, suffix)
    except Exception:
        try:
            return _render_with_trimesh(scene, output_dir, resolution, camera_pose, suffix)
        except Exception:
            meshes = _collect_meshes(scene)
            combined = trimesh.util.concatenate(meshes)
            return _render_silhouette(combined, output_dir, resolution, azimuth, elevation, suffix)


def extract_silhouette_mask(image_path: str, size: tuple = (256, 256)) -> np.ndarray:
    """
    Extract a binary foreground silhouette mask (0/255) from an image.

    Uses the alpha channel when present, otherwise thresholds against a
    near-white/near-black background.
    """
    import cv2

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    if img.ndim == 3 and img.shape[-1] == 4:
        mask = (img[:, :, 3] > 128).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        # Foreground = pixels that differ from a white OR black background.
        not_white = gray < 240
        not_black = gray > 15
        mask = (not_white & not_black).astype(np.uint8) * 255
        # If thresholding caught almost nothing (e.g. pure shapes), fall back to non-white.
        if mask.sum() < gray.size * 0.005:
            mask = (gray < 240).astype(np.uint8) * 255

    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return mask


# ──────────────── Camera math ────────────────


def _look_at(eye: np.ndarray, target: np.ndarray, up=np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """Build a camera-to-world matrix (OpenGL convention: camera looks down -Z)."""
    forward = eye - target
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        forward = np.array([0.0, 0.0, 1.0])
    else:
        forward = forward / norm

    right = np.cross(up, forward)
    rnorm = np.linalg.norm(right)
    if rnorm < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / rnorm

    true_up = np.cross(forward, right)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def _camera_pose_for(scene, azimuth: float, elevation: float) -> np.ndarray:
    """Compute a camera-to-world pose orbiting the scene center."""
    bounds = scene.bounds
    center = (bounds[0] + bounds[1]) / 2
    size = float(np.linalg.norm(bounds[1] - bounds[0]))
    distance = max(size * 1.5, 1e-3)

    az = np.radians(azimuth)
    el = np.radians(elevation)
    direction = np.array([
        np.cos(el) * np.sin(az),
        np.sin(el),
        np.cos(el) * np.cos(az),
    ])
    eye = center + distance * direction
    return _look_at(eye, center)


# ──────────────── Renderers ────────────────


def _collect_meshes(scene):
    meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
    if not meshes:
        raise ValueError("No valid meshes found in GLB file")
    return meshes


def _save_rgba(color, output_dir: str, suffix: str) -> str:
    file_id = str(uuid.uuid4())
    output_path = os.path.join(output_dir, f"{file_id}_{suffix}.png")
    Image.fromarray(color).save(output_path)
    return output_path


def _render_with_pyrender(scene, output_dir, resolution, camera_pose, suffix) -> str:
    """Render using PyRender with an offscreen renderer and an RGBA buffer."""
    import pyrender

    py_scene = pyrender.Scene.from_trimesh_scene(scene, bg_color=[0, 0, 0, 0])

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    py_scene.add(light, pose=camera_pose)
    ambient = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    py_scene.add(ambient)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    py_scene.add(camera, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(*resolution)
    color, _ = renderer.render(py_scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    return _save_rgba(color, output_dir, suffix)


def _render_with_trimesh(scene, output_dir, resolution, camera_pose, suffix) -> str:
    """Fallback renderer using Trimesh's built-in rendering at a given pose."""
    meshes = _collect_meshes(scene)
    combined = trimesh.util.concatenate(meshes)
    render_scene = trimesh.Scene(combined)

    try:
        render_scene.camera_transform = camera_pose
    except Exception:
        pass

    png_data = render_scene.save_image(resolution=resolution)
    file_id = str(uuid.uuid4())
    output_path = os.path.join(output_dir, f"{file_id}_{suffix}.png")
    with open(output_path, "wb") as f:
        f.write(png_data)
    return output_path


def _render_silhouette(mesh, output_dir, resolution, azimuth, elevation, suffix) -> str:
    """Last-resort silhouette: project vertices after orbiting the mesh."""
    import cv2

    vertices = mesh.vertices - mesh.vertices.mean(axis=0)

    # Rotate vertices by -azimuth (Y) and -elevation (X) to emulate the camera orbit.
    az = np.radians(-azimuth)
    el = np.radians(-elevation)
    ry = np.array([[np.cos(az), 0, np.sin(az)], [0, 1, 0], [-np.sin(az), 0, np.cos(az)]])
    rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
    vertices = vertices @ ry.T @ rx.T

    max_extent = np.abs(vertices).max()
    if max_extent > 0:
        vertices = vertices / max_extent

    w, h = resolution
    margin = 0.1
    scale = min(w, h) * (1 - 2 * margin) / 2

    img = np.zeros((h, w, 4), dtype=np.uint8)  # transparent background

    pts = vertices[:, :2]
    px = (pts[:, 0] * scale + w / 2).astype(np.int32)
    py = (h / 2 - pts[:, 1] * scale).astype(np.int32)

    for face in mesh.faces:
        tri = np.stack([px[face], py[face]], axis=1)
        cv2.fillConvexPoly(img, tri, (160, 160, 160, 255))

    return _save_rgba(img, output_dir, suffix)
