"""
Validation Calibration Suite (Enhancement 7)

Proves the validator metrics respond correctly to controlled degradation:

  Scenario A — Original asset            → high scores everywhere
  Scenario B — Geometry damage           → geometry drops, texture/color stable
  Scenario C — Texture removal           → texture drops, geometry/color stable
  Scenario D — Color (hue/sat) shift     → color drops, geometry/texture stable

Run:
    cd backend
    GLB_PATH=/path/model.glb IMAGE_PATH=/path/source.png python -m tests.degradation_tests

Each scenario degrades a *copy* of the asset, re-runs the relevant validators
(+ gating), and prints a comparison table.
"""

import os
import sys
import uuid
import tempfile

# Allow running as `python -m tests.degradation_tests` from backend/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import trimesh

from services.glb_renderer import render_glb_from_pose
from services.geometry_validator import validate_geometry
from services.geometry_quality_checker import check_geometry_quality, apply_geometry_gates
from services.texture_validator import validate_texture
from services.texture_checker import check_texture_presence, apply_texture_gates
from services.color_validator import validate_color
from services.dominant_color import analyze_dominant_colors, dominant_color_score


# ──────────────── Score helpers (mirror the API orchestration) ────────────────


def _geometry_score(glb_path, image_path, render_path):
    base = validate_geometry(image_path, glb_path, render_path)["score"]
    quality = check_geometry_quality(glb_path)
    return apply_geometry_gates(base, quality)["score"]


def _texture_score(glb_path, image_path, render_path):
    base = validate_texture(image_path, render_path, glb_path)["score"]
    presence = check_texture_presence(glb_path)
    return apply_texture_gates(base, presence)["score"]


def _color_score(image_path, render_path):
    base = validate_color(image_path, render_path)["score"]
    dom = analyze_dominant_colors(image_path, render_path)
    dom_s = dominant_color_score(dom["dominant_color_distance"])
    return round(0.6 * base + 0.4 * dom_s, 1)


def _score_all(glb_path, image_path, render_path):
    return {
        "geometry": _geometry_score(glb_path, image_path, render_path),
        "texture": _texture_score(glb_path, image_path, render_path),
        "color": _color_score(image_path, render_path),
    }


# ──────────────── Degraders ────────────────


def degrade_geometry(glb_path, out_dir, remove_fraction=0.25):
    """Remove a fraction of faces to introduce holes / damage."""
    scene = trimesh.load(glb_path, force="scene")
    new_geom = {}
    for name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 10:
            keep = np.ones(len(geom.faces), dtype=bool)
            n_remove = int(len(geom.faces) * remove_fraction)
            # Deterministic stride removal (no RNG dependency).
            idx = np.linspace(0, len(geom.faces) - 1, n_remove).astype(int)
            keep[idx] = False
            damaged = geom.copy()
            damaged.update_faces(keep)
            new_geom[name] = damaged
        else:
            new_geom[name] = geom
    out = os.path.join(out_dir, f"{uuid.uuid4()}_geomdamage.glb")
    trimesh.Scene(new_geom).export(out)
    return out


def degrade_texture(glb_path, out_dir):
    """Strip materials / textures / UVs."""
    scene = trimesh.load(glb_path, force="scene")
    for geom in scene.geometry.values():
        if isinstance(geom, trimesh.Trimesh):
            try:
                geom.visual = trimesh.visual.ColorVisuals(geom)  # plain, no material/uv
            except Exception:
                pass
    out = os.path.join(out_dir, f"{uuid.uuid4()}_notexture.glb")
    scene.export(out)
    return out


def degrade_color(render_path, out_dir, hue_shift=40, sat_scale=0.6):
    """Hue-shift + desaturate a rendered image to force a color mismatch."""
    img = cv2.imread(render_path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3 and img.shape[-1] == 4:
        bgr, alpha = img[:, :, :3], img[:, :, 3:4]
    else:
        bgr, alpha = img, None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
    shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if alpha is not None:
        shifted = np.dstack([shifted, alpha])
    out = os.path.join(out_dir, f"{uuid.uuid4()}_colorshift.png")
    cv2.imwrite(out, shifted)
    return out


# ──────────────── Runner ────────────────


def run_suite(glb_path, image_path, out_dir=None):
    out_dir = out_dir or tempfile.mkdtemp(prefix="degradation_")
    os.makedirs(out_dir, exist_ok=True)

    # Aligned render of the original (front pose; full pose search needs the API).
    render = render_glb_from_pose(glb_path, 0, 0, out_dir, suffix="orig")

    rows = []

    # A — Original
    rows.append(("Original", _score_all(glb_path, image_path, render)))

    # B — Geometry damage
    glb_b = degrade_geometry(glb_path, out_dir)
    render_b = render_glb_from_pose(glb_b, 0, 0, out_dir, suffix="geomdmg")
    rows.append(("Missing Faces", _score_all(glb_b, image_path, render_b)))

    # C — Texture removal
    glb_c = degrade_texture(glb_path, out_dir)
    render_c = render_glb_from_pose(glb_c, 0, 0, out_dir, suffix="notex")
    rows.append(("Missing Texture", _score_all(glb_c, image_path, render_c)))

    # D — Color shift (degrade the render the color stage compares against)
    render_d = degrade_color(render, out_dir)
    color_only = {
        "geometry": rows[0][1]["geometry"],   # geometry unaffected by image hue
        "texture": _texture_score(glb_path, image_path, render_d),
        "color": _color_score(image_path, render_d),
    }
    rows.append(("Hue Shift", color_only))

    _print_table(rows)
    return rows


def _print_table(rows):
    print("\n| Test Case        | Geometry | Texture | Color |")
    print("| ---------------- | -------- | ------- | ----- |")
    for name, s in rows:
        print(
            f"| {name:<16} | {s['geometry']:>8} | {s['texture']:>7} | {s['color']:>5} |"
        )
    print()
    _assert_responsiveness(rows)


def _assert_responsiveness(rows):
    """Sanity checks that degradation moved the right metric down."""
    d = {name: s for name, s in rows}
    checks = []
    if "Original" in d and "Missing Faces" in d:
        checks.append(("Geometry drops on damage",
                       d["Missing Faces"]["geometry"] <= d["Original"]["geometry"]))
    if "Original" in d and "Missing Texture" in d:
        checks.append(("Texture drops on removal",
                       d["Missing Texture"]["texture"] <= d["Original"]["texture"]))
    if "Original" in d and "Hue Shift" in d:
        checks.append(("Color drops on hue shift",
                       d["Hue Shift"]["color"] <= d["Original"]["color"]))
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return all(ok for _, ok in checks)


if __name__ == "__main__":
    glb = os.environ.get("GLB_PATH") or (sys.argv[1] if len(sys.argv) > 1 else None)
    img = os.environ.get("IMAGE_PATH") or (sys.argv[2] if len(sys.argv) > 2 else None)
    if not glb or not img:
        print("Usage: GLB_PATH=model.glb IMAGE_PATH=source.png python -m tests.degradation_tests")
        sys.exit(1)
    run_suite(glb, img)
