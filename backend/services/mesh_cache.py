"""
GLB Mesh Cache

A single validation request touches the same GLB from six different services
(pose render, geometry integrity, geometry quality, texture presence x2,
texture render). Loading it once per call expands a 100 MB GLB into 1.5-2 GB of
Python/C structures — six times — producing transient 6-12 GB spikes.

This module loads each GLB exactly once and hands the same in-memory
`trimesh.Scene` (and a cached concatenated `Trimesh`) to every consumer. The
cache is keyed by (path, mtime, size) and holds only the most recently loaded
model, so memory stays bounded across requests while a single request reuses
one load.
"""

import os
import threading

import trimesh

_lock = threading.Lock()
_scene_cache = {}      # key -> trimesh.Scene
_combined_cache = {}   # key -> trimesh.Trimesh | None


def _key(glb_path: str):
    try:
        st = os.stat(glb_path)
        return (os.path.abspath(glb_path), st.st_mtime_ns, st.st_size)
    except OSError:
        return (os.path.abspath(glb_path), None, None)


def load_scene(glb_path: str):
    """Return a cached `trimesh.Scene` for the GLB, loading it once."""
    key = _key(glb_path)
    with _lock:
        scene = _scene_cache.get(key)
        if scene is None:
            scene = trimesh.load(glb_path, force="scene")
            # Keep only the most recent model to bound memory across requests.
            _scene_cache.clear()
            _combined_cache.clear()
            _scene_cache[key] = scene
        return scene


def load_combined(glb_path: str):
    """
    Return a cached concatenated `Trimesh` of every mesh in the scene, or
    None if the GLB contains no valid meshes. Concatenation is shared across
    every service that needs the combined mesh.
    """
    key = _key(glb_path)
    scene = load_scene(glb_path)
    with _lock:
        if key in _combined_cache:
            return _combined_cache[key]
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        combined = trimesh.util.concatenate(meshes) if meshes else None
        _combined_cache[key] = combined
        return combined
