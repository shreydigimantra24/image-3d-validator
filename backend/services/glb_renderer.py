"""
GLB Renderer Service — renders a GLB model to a 2D image using Trimesh + PyRender.

Supports rendering from arbitrary camera poses (azimuth/elevation) so the
pose estimator can search for the viewpoint that best matches the input image.
"""

import os
import uuid
import logging
import numpy as np
import trimesh
from PIL import Image

from services.mesh_cache import load_scene

logger = logging.getLogger(__name__)

# Default vertical field of view (degrees). The pose estimator may override this
# per-render once it has searched for the FOV that best matches the photo.
DEFAULT_FOV_DEG = 60.0


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
    distance: float = None,
    fov_deg: float = DEFAULT_FOV_DEG,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
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
        distance: Camera distance from scene center. None → auto (size * 1.5).
        fov_deg: Vertical field of view in degrees.
        offset_x, offset_y: Camera pan in its right/up plane (world units) — used
            to translate the rendered object so it lines up with the photo.

    Returns:
        Path to the rendered PNG image.
    """
    scene = load_scene(glb_path)
    camera_pose = _camera_pose_for(
        scene, azimuth, elevation, distance=distance,
        offset_x=offset_x, offset_y=offset_y,
    )

    try:
        return _render_with_pyrender(scene, output_dir, resolution, camera_pose, suffix, fov_deg)
    except Exception:
        logger.exception("pyrender render failed; falling back to trimesh")
        try:
            return _render_with_trimesh(scene, output_dir, resolution, camera_pose, suffix)
        except Exception:
            logger.exception("trimesh render failed; falling back to analytic silhouette")
            meshes = _collect_meshes(scene)
            combined = trimesh.util.concatenate(meshes)
            return _render_silhouette(
                combined, output_dir, resolution, camera_pose, fov_deg, suffix
            )


class PoseRenderer:
    """
    Reusable renderer for pose search. Loads the GLB once and reuses a single
    pyrender scene + offscreen renderer across all candidate viewpoints,
    returning silhouette masks in memory (no per-candidate file writes).

    This is the memory-safe path for scanning dozens of poses: creating a fresh
    renderer and reloading the mesh per candidate is what blows up RAM.

    Use as a context manager:
        with PoseRenderer(glb_path) as pr:
            mask = pr.mask_at(azimuth, elevation)
    """

    def __init__(self, glb_path: str, resolution: tuple = (256, 256), fov_deg: float = DEFAULT_FOV_DEG):
        self.resolution = resolution
        self.fov_deg = fov_deg
        self.scene = load_scene(glb_path)

        # Cache the geometry framing so the estimator can seed its distance
        # search and so the silhouette fallback can build a correct camera.
        bounds = self.scene.bounds
        self.center = (bounds[0] + bounds[1]) / 2
        self.scene_size = float(np.linalg.norm(bounds[1] - bounds[0]))
        self.default_distance = max(self.scene_size * 1.5, 1e-3)

        self._mode = None
        self._pyrender = None
        self._py_scene = None
        self._camera = None
        self._cam_node = None
        self._light_node = None
        self._renderer = None
        self._tri_scene = None
        self._combined = None
        self._setup()

    def _setup(self):
        # Preferred: a single reusable pyrender offscreen renderer.
        #
        # Pose search only needs the SILHOUETTE (alpha mask), never the
        # texture. Build the pyrender scene from geometry-only meshes so no
        # texture is uploaded to the GL context. This is both faster and avoids
        # the PyOpenGL glGenTextures crash that breaks textured uploads on some
        # PyOpenGL/numpy combinations — which would otherwise fail every
        # candidate render and force a silent fallback.
        try:
            import pyrender

            self._pyrender = pyrender
            self._py_scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
            for geom in _collect_meshes(self.scene):
                bare = trimesh.Trimesh(
                    vertices=geom.vertices, faces=geom.faces, process=False
                )
                self._py_scene.add(pyrender.Mesh.from_trimesh(bare, smooth=False))
            self._camera = pyrender.PerspectiveCamera(yfov=np.radians(self.fov_deg))
            self._cam_node = self._py_scene.add(self._camera)
            self._light_node = self._py_scene.add(
                pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
            )
            self._renderer = pyrender.OffscreenRenderer(*self.resolution)
            # Render once now so a GL failure surfaces here (and we fall back to
            # the analytic path) instead of silently failing every candidate.
            pose = _camera_pose_for(self.scene, 0, 0)
            self._mask_pyrender(pose, self.fov_deg)
            self._mode = "pyrender"
            return
        except Exception:
            logger.exception("PoseRenderer pyrender setup failed; using silhouette projection")
            self._teardown_pyrender()

        # Fallback: analytic silhouette projection (pure numpy/cv2, no GL).
        self._combined = trimesh.util.concatenate(_collect_meshes(self.scene))
        self._mode = "silhouette"

    def mask_at(
        self,
        azimuth: float,
        elevation: float,
        distance: float = None,
        fov_deg: float = None,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> np.ndarray:
        """
        Return a binary (0/255) silhouette mask at the given pose.

        distance/fov_deg default to the renderer's auto distance / current FOV.
        offset_x/offset_y pan the camera so the silhouette can be translated to
        match the photo (used by the alignment optimizers).
        """
        fov_deg = self.fov_deg if fov_deg is None else fov_deg
        pose = _camera_pose_for(
            self.scene, azimuth, elevation, distance=distance,
            offset_x=offset_x, offset_y=offset_y,
        )
        if self._mode == "pyrender":
            return self._mask_pyrender(pose, fov_deg)
        return self._mask_silhouette(pose, fov_deg)

    def _mask_pyrender(self, pose, fov_deg) -> np.ndarray:
        # Update the FOV live so the estimator can search candidate FOVs without
        # rebuilding the scene/renderer.
        if abs(self._camera.yfov - np.radians(fov_deg)) > 1e-6:
            self._camera.yfov = np.radians(fov_deg)
        self._py_scene.set_pose(self._cam_node, pose)
        self._py_scene.set_pose(self._light_node, pose)
        color, _ = self._renderer.render(self._py_scene, flags=self._pyrender.RenderFlags.RGBA)
        alpha = color[:, :, 3]
        return (alpha > 16).astype(np.uint8) * 255

    def _mask_silhouette(self, pose, fov_deg) -> np.ndarray:
        """Pure-numpy perspective silhouette matching the pyrender camera so the
        distance/fov/offset optimizers behave identically on the fallback path."""
        import cv2

        w, h = self.resolution
        px, py, depth = _project_points(self._combined.vertices, pose, fov_deg, w, h)
        mask = np.zeros((h, w), dtype=np.uint8)
        ix = px.astype(np.int32)
        iy = py.astype(np.int32)
        for face in self._combined.faces:
            if np.any(depth[face] <= 0):
                continue  # face (partly) behind the camera
            tri = np.stack([ix[face], iy[face]], axis=1)
            cv2.fillConvexPoly(mask, tri, 255)
        return mask

    def _teardown_pyrender(self):
        try:
            if self._renderer is not None:
                self._renderer.delete()
        except Exception:
            pass
        self._renderer = None
        self._py_scene = None

    def close(self):
        self._teardown_pyrender()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


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


def _camera_pose_for(
    scene,
    azimuth: float,
    elevation: float,
    distance: float = None,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> np.ndarray:
    """
    Compute a camera-to-world pose orbiting the scene center.

    distance: explicit camera distance (None → auto, size * 1.5).
    offset_x/offset_y: pan the camera in its own right/up plane (world units).
        This implements ``look_at(center + offset)`` as a pure translation, so
        the silhouette slides across the frame without changing orientation —
        used to centre the render on the photographed object.
    """
    bounds = scene.bounds
    center = (bounds[0] + bounds[1]) / 2
    size = float(np.linalg.norm(bounds[1] - bounds[0]))
    if distance is None:
        distance = max(size * 1.5, 1e-3)

    az = np.radians(azimuth)
    el = np.radians(elevation)
    direction = np.array([
        np.cos(el) * np.sin(az),
        np.sin(el),
        np.cos(el) * np.cos(az),
    ])
    eye = center + distance * direction
    pose = _look_at(eye, center)

    if offset_x or offset_y:
        right = pose[:3, 0]
        up = pose[:3, 1]
        pan = right * offset_x + up * offset_y
        pose[:3, 3] = eye + pan  # translate camera; orientation unchanged
    return pose


def _project_points(verts: np.ndarray, camera_pose: np.ndarray, fov_deg: float, w: int, h: int):
    """
    Perspective-project world vertices through a camera-to-world pose, matching
    pyrender's pinhole + OpenGL convention (camera looks down its local -Z).

    Returns (px, py, depth) where depth > 0 is in front of the camera.
    """
    R = camera_pose[:3, :3]
    eye = camera_pose[:3, 3]
    rel = verts - eye
    x = rel @ R[:, 0]            # right component
    y = rel @ R[:, 1]            # up component
    depth = -(rel @ R[:, 2])     # forward column points behind → negate
    f = (h / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    z = np.maximum(depth, 1e-6)
    px = x / z * f + w / 2.0
    py = h / 2.0 - y / z * f
    return px, py, depth


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


def _render_with_pyrender(scene, output_dir, resolution, camera_pose, suffix, fov_deg=DEFAULT_FOV_DEG) -> str:
    """Render using PyRender with an offscreen renderer and an RGBA buffer."""
    import pyrender

    py_scene = pyrender.Scene.from_trimesh_scene(scene, bg_color=[0, 0, 0, 0])

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    py_scene.add(light, pose=camera_pose)
    ambient = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    py_scene.add(ambient)

    camera = pyrender.PerspectiveCamera(yfov=np.radians(fov_deg))
    py_scene.add(camera, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(*resolution)
    try:
        color, _ = renderer.render(py_scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        # Guarantee the EGL/OpenGL context is released even if render() raises,
        # otherwise the context is orphaned and leaks VRAM on headless servers.
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


def _render_silhouette(mesh, output_dir, resolution, camera_pose, fov_deg, suffix) -> str:
    """Last-resort silhouette: perspective-project the mesh through the same
    camera pose / FOV the optimizer chose, so distance, offset and FOV still
    take effect on the saved render."""
    import cv2

    w, h = resolution
    px, py, depth = _project_points(mesh.vertices, camera_pose, fov_deg, w, h)
    ix = px.astype(np.int32)
    iy = py.astype(np.int32)

    img = np.zeros((h, w, 4), dtype=np.uint8)  # transparent background
    for face in mesh.faces:
        if np.any(depth[face] <= 0):
            continue
        tri = np.stack([ix[face], iy[face]], axis=1)
        cv2.fillConvexPoly(img, tri, (160, 160, 160, 255))

    return _save_rgba(img, output_dir, suffix)
