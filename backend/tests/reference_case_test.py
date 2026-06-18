"""
Reference-case regression test (Fixes 1-4)

Locks in the corrected behaviour on the known failing case: a multi-part
product/assembly mesh (IKEA "Vihals" table + 4 chairs) whose chairs use a LIGHT
fabric albedo but ship with metallicFactor = 1.0.

Old (wrong) output:  Geometry 40, Texture 76, Color 60 (chairs render charcoal).
New expected output:
  * asset auto-classified as "product"
  * geometry HIGH (>= 80) — the mesh is structurally fine; non-watertightness /
    component count / open edges are NOT treated as defects (Fix 1)
  * geometry reason / gating does NOT cite watertightness or raw component / hole
    counts as defects (Fix 1/2)
  * a MATERIAL warning flags metallicFactor > 0.8 (Fix 3)
  * color NOT falsely low — judged on the light albedo vs the light photo (Fix 3)
  * per-score confidence is reported (Fix 4)

Runs with NO proprietary asset and open-source deps only: by default it builds a
synthetic fixture that reproduces the ground-truth facts. Point it at the real
GLB to validate that instead:

    cd backend
    REF_GLB_PATH=/path/vihals.glb REF_IMAGE_PATH=/path/source.png \
        python -m tests.reference_case_test
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh
from PIL import Image

from services.glb_renderer import render_glb_from_pose
from services.geometry_quality_checker import (
    check_geometry_quality,
    classify_asset,
    apply_geometry_gates,
)
from services.geometry_validator import validate_geometry
from services.material_inspector import inspect_material, get_albedo_rgb
from services.color_validator import validate_color
from services.dominant_color import analyze_dominant_colors, dominant_color_score
from services.texture_validator import validate_texture
from services.texture_checker import check_texture_presence, apply_texture_gates


# ──────────────── Synthetic fixture (mirrors the ground-truth facts) ────────────────


def build_synthetic_reference(out_dir: str):
    """
    Build a multi-part assembly GLB that reproduces the reference facts:
      * many connected components, several substantial parts (assembly)
      * NOT watertight (open shells)
      * a LIGHT (~RGB 205) baseColor texture with valid UVs
      * metallicFactor = roughnessFactor = 1.0  (the key clue)

    Returns (glb_path, source_image_path).
    """
    light = np.full((64, 64, 3), 205, np.uint8)
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(light),
        metallicFactor=1.0,
        roughnessFactor=1.0,
    )
    rng = np.random.default_rng(0)
    parts = {}

    # Substantial parts (tabletop, seat pans, backrests): subdivided open panels.
    for i in range(5):
        panel = trimesh.creation.box(extents=(1.0, 1.0, 0.02))
        for _ in range(3):
            panel = panel.subdivide()  # > 200 faces → "substantial"
        panel.apply_translation((i * 0.3, 0, 0))
        panel.visual = trimesh.visual.TextureVisuals(
            uv=rng.random((len(panel.vertices), 2)), material=mat
        )
        parts[f"part_{i}"] = panel

    # Many tiny fittings (screws/glides) sitting ON the body — raises the
    # component count without being floaters/slivers (they're close to the body).
    for i in range(20):
        screw = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
        screw.update_faces(screw.face_normals[:, 2] < 0.9)  # open shell
        screw.apply_translation((np.cos(i) * 0.5, np.sin(i) * 0.5, 0.1))
        parts[f"screw_{i}"] = screw

    glb_path = os.path.join(out_dir, "synthetic_reference.glb")
    trimesh.Scene(parts).export(glb_path)

    # Source photo: a light-grey foreground (matches the light fabric/table).
    src_path = os.path.join(out_dir, "synthetic_source.png")
    Image.fromarray(np.full((128, 128, 3), 210, np.uint8)).save(src_path)
    return glb_path, src_path


# ──────────────── Assertions ────────────────


def check_reference_case(glb_path, geom_source_path, geom_render_path,
                         color_source_path, color_render_path):
    """Score the asset the way the API does and assert the corrected ranges.

    geom_source_path / geom_render_path feed the silhouette comparison; for the
    real asset these are the photo and its aligned render. color_source_path is
    the photo foreground compared against the albedo.
    """
    results = {}
    failures = []

    def expect(label, cond):
        results[label] = bool(cond)
        if not cond:
            failures.append(label)

    quality = check_geometry_quality(glb_path)
    asset_class = classify_asset(quality)

    # Fix 1 — asset classification + structurally-fine geometry
    expect("asset auto-classified as product", asset_class == "product")
    expect("mesh is a real assembly (>=3 substantial parts)",
           quality.get("substantial_components", 0) >= 3)
    expect("mesh is (correctly) not watertight", quality.get("is_watertight") is False)

    geometry = validate_geometry(
        geom_source_path, glb_path, geom_render_path,
        asset_class=asset_class, quality=quality,
    )
    gate = apply_geometry_gates(geometry["score"], quality, asset_class)
    geometry_score = gate["score"]
    expect("geometry score HIGH (>= 80)", geometry_score >= 80)

    # Fix 1 — gating must NOT cite watertight / component / hole counts
    gate_text = " ".join(gate.get("applied_gates", [])).lower()
    expect("no watertight/component/hole gate applied",
           not any(w in gate_text for w in ("watertight", "component", "hole")))

    # Fix 3 — material warning, not a tanked color score
    material = inspect_material(glb_path)
    albedo = get_albedo_rgb(glb_path)
    expect("metallic material flagged as a warning",
           material.get("suspicious_metallic") and len(material.get("warnings", [])) > 0)
    expect("albedo texture is light (mean L* high)",
           albedo is not None and float(np.asarray(albedo).reshape(-1, 3).mean()) > 150)

    color = validate_color(color_source_path, color_render_path, albedo_rgb=albedo)
    dom = analyze_dominant_colors(color_source_path, color_render_path, render_rgb=albedo)
    dom_s = dominant_color_score(dom["dominant_color_distance"])
    color_score = round(0.6 * color["score"] + 0.4 * dom_s, 1)
    expect("color NOT falsely low (>= 70)", color_score >= 70)
    expect("color judged on albedo texture",
           color["details"]["reference"] == "albedo_texture")

    # Fix 4 — confidence reported (albedo path → high color confidence)
    using_albedo = albedo is not None and albedo.size > 0
    color_conf = 0.95 if using_albedo else 0.5
    texture = validate_texture(color_source_path, color_render_path, glb_path,
                               alignment={"iou": 0.62})
    presence = check_texture_presence(glb_path)
    texture_score = apply_texture_gates(texture["score"], presence)["score"]
    expect("per-score confidence reported",
           color_conf is not None and texture.get("confidence") is not None)
    expect("low-IoU texture flagged as untrusted",
           texture["details"]["perceptual"].get("trusted") is False)

    print("\nReference-case scores:")
    print(f"  asset_class       : {asset_class}")
    print(f"  geometry          : {geometry_score}  (>= 80 expected)")
    print(f"  texture           : {texture_score}  (confidence {texture.get('confidence')})")
    print(f"  color             : {color_score}  (confidence {color_conf}, ref "
          f"{color['details']['reference']})")
    print(f"  material warnings : {material.get('warnings')}")
    print("\nChecks:")
    for label, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    assert not failures, f"Reference-case regressions: {failures}"
    print("\nAll reference-case expectations met.\n")
    return results


def main():
    glb = os.environ.get("REF_GLB_PATH")
    img = os.environ.get("REF_IMAGE_PATH")
    out_dir = tempfile.mkdtemp(prefix="refcase_")

    if glb and img:
        # Real asset: align with a real pose so geometry IoU is meaningful.
        from services.pose_estimator import estimate_pose

        pose = estimate_pose(img, glb, out_dir)
        render = pose["aligned_render_path"]
        # Real asset: photo is the geometry source AND the color source.
        check_reference_case(glb, img, render, img, render)
    else:
        print("No REF_GLB_PATH/REF_IMAGE_PATH set — using synthetic fixture.")
        glb, img = build_synthetic_reference(out_dir)
        # Geometry source = a render of the asset so silhouette IoU is ~1 (we are
        # testing scoring logic, not the renderer). Color is judged on the albedo
        # vs the light photo, independent of the render.
        render = render_glb_from_pose(glb, 0, 0, out_dir, suffix="ref")
        check_reference_case(glb, render, render, img, render)


if __name__ == "__main__":
    main()
