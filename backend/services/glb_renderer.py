"""
GLB Renderer Service — renders a GLB model to a 2D image using Trimesh + PyRender.
"""

import os
import uuid
import numpy as np
import trimesh
from PIL import Image


def render_glb(glb_path: str, output_dir: str, resolution: tuple = (512, 512)) -> str:
    """
    Load a GLB file and render it to a 2D PNG image.

    Uses Trimesh's built-in scene rendering. Falls back to a simple
    projection if pyrender/OpenGL is unavailable (headless servers).

    Args:
        glb_path: Path to the GLB file.
        output_dir: Directory to save the rendered image.
        resolution: Output image resolution (width, height).

    Returns:
        Path to the rendered PNG image.
    """
    # Load the GLB as a trimesh Scene
    scene = trimesh.load(glb_path, force="scene")

    try:
        # Attempt PyRender-based rendering (requires OpenGL)
        rendered_path = _render_with_pyrender(scene, glb_path, output_dir, resolution)
    except Exception:
        # Fallback: use trimesh's built-in renderer
        rendered_path = _render_with_trimesh(scene, output_dir, resolution)

    return rendered_path


def _render_with_pyrender(scene, glb_path: str, output_dir: str, resolution: tuple) -> str:
    """Render using PyRender with offscreen renderer."""
    import pyrender

    # Convert trimesh scene to pyrender scene
    py_scene = pyrender.Scene.from_trimesh_scene(scene)

    # Add directional light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    py_scene.add(light)

    # Add ambient light
    ambient = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    py_scene.add(ambient)

    # Set up camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

    # Compute camera position from scene bounds
    bounds = scene.bounds
    center = (bounds[0] + bounds[1]) / 2
    size = np.linalg.norm(bounds[1] - bounds[0])
    camera_distance = size * 1.5

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = center + np.array([0, 0, camera_distance])
    py_scene.add(camera, pose=camera_pose)

    # Offscreen rendering
    renderer = pyrender.OffscreenRenderer(*resolution)
    color, _ = renderer.render(py_scene)
    renderer.delete()

    # Save
    file_id = str(uuid.uuid4())
    output_filename = f"{file_id}_rendered.png"
    output_path = os.path.join(output_dir, output_filename)
    Image.fromarray(color).save(output_path)

    return output_path


def _render_with_trimesh(scene, output_dir: str, resolution: tuple) -> str:
    """Fallback renderer using Trimesh's built-in rendering."""
    # Get all meshes from scene
    meshes = []
    for name, geometry in scene.geometry.items():
        if isinstance(geometry, trimesh.Trimesh):
            meshes.append(geometry)

    if not meshes:
        raise ValueError("No valid meshes found in GLB file")

    # Combine all meshes
    combined = trimesh.util.concatenate(meshes)

    # Create a scene with proper camera
    render_scene = trimesh.Scene(combined)

    # Try to export an image using the scene
    try:
        png_data = render_scene.save_image(resolution=resolution)
        file_id = str(uuid.uuid4())
        output_filename = f"{file_id}_rendered.png"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "wb") as f:
            f.write(png_data)
        return output_path
    except Exception:
        # Last resort: create a silhouette from orthographic projection
        return _render_silhouette(combined, output_dir, resolution)


def _render_silhouette(mesh, output_dir: str, resolution: tuple) -> str:
    """Create a simple 2D silhouette by projecting vertices."""
    vertices = mesh.vertices
    # Center and normalize
    center = vertices.mean(axis=0)
    vertices = vertices - center
    max_extent = np.abs(vertices).max()
    if max_extent > 0:
        vertices = vertices / max_extent

    # Project to 2D (XY plane)
    w, h = resolution
    margin = 0.1
    scale = min(w, h) * (1 - 2 * margin) / 2

    img = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background

    # Draw faces
    for face in mesh.faces:
        pts = vertices[face][:, :2]  # Project to XY
        px = (pts[:, 0] * scale + w / 2).astype(int)
        py = (h / 2 - pts[:, 1] * scale).astype(int)

        # Simple triangle fill would require cv2, so just draw vertices
        for x, y in zip(px, py):
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = [128, 128, 128]

    file_id = str(uuid.uuid4())
    output_filename = f"{file_id}_rendered.png"
    output_path = os.path.join(output_dir, output_filename)
    Image.fromarray(img).save(output_path)

    return output_path
