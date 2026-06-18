"""
Validate API — orchestrates the full quality-gate pipeline:

  1. Silhouette extraction + camera pose search  (Enhancement 1)
  2. Geometry validation + structural gating      (Enhancement 3)
  3. Texture validation + presence gating         (Enhancement 4)
  4. Color validation + dominant-color analysis   (Enhancement 5)
  5. Score aggregation
  6. Reason generation
  7. Validation evidence overlay                   (Enhancement 2)

Per-stage latency is tracked throughout (Enhancement 6).
"""

import os
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from services.pose_estimator import estimate_pose
from services.geometry_validator import validate_geometry
from services.geometry_quality_checker import (
    check_geometry_quality,
    apply_geometry_gates,
    classify_asset,
)
from services.texture_validator import validate_texture
from services.texture_checker import check_texture_presence, apply_texture_gates
from services.color_validator import validate_color
from services.dominant_color import analyze_dominant_colors, dominant_color_score
from services.material_inspector import inspect_material, get_albedo_rgb
from services.evidence_generator import generate_overlay
from services.reason_generator import generate_reasons
from services.validation_config import IOU_TRUST_THRESHOLD

router = APIRouter()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ValidateRequest(BaseModel):
    image_path: str                              # bg-removed image
    glb_path: str                                # GLB model
    original_image_path: Optional[str] = None    # original image (optional)
    # "product" | "single_solid" | None (auto-detect). Defaults to auto-detect,
    # which treats multi-part assembly meshes (furniture) as products so they
    # aren't falsely penalised for non-watertightness / open shells (Fix 1).
    asset_class: Optional[str] = None


@router.post("/validate")
async def validate_model(request: ValidateRequest):
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Preprocessed image not found")
    if not os.path.exists(request.glb_path):
        raise HTTPException(status_code=404, detail="GLB model not found")

    timings = {}
    t_start = time.perf_counter()

    try:
        # ── Step 1: Camera pose alignment ──
        t0 = time.perf_counter()
        pose = estimate_pose(request.image_path, request.glb_path, OUTPUT_DIR)
        aligned_render_path = pose["aligned_render_path"]
        align_iou = pose.get("iou")
        align_trusted = align_iou is not None and float(align_iou) >= IOU_TRUST_THRESHOLD
        timings["alignment"] = round(time.perf_counter() - t0, 3)

        # ── Step 2: Geometry validation (asset-class aware, render-first) ──
        t0 = time.perf_counter()
        quality = check_geometry_quality(request.glb_path)
        asset_class = classify_asset(quality, request.asset_class)
        geometry_result = validate_geometry(
            source_image_path=request.image_path,
            glb_path=request.glb_path,
            rendered_image_path=aligned_render_path,
            asset_class=asset_class,
            quality=quality,
        )
        # Asset-aware gate: products are NOT capped for watertightness / component
        # count / open edges — only for genuine defects (Fix 1).
        gate_g = apply_geometry_gates(geometry_result["score"], quality, asset_class)
        geometry_score = gate_g["score"]
        geometry_result["details"]["quality_checks"] = quality
        geometry_result["details"]["gating"] = gate_g
        # Geometry confidence tracks how well the rendered shape aligned.
        geometry_conf = round(float(align_iou), 3) if align_iou is not None else 0.0
        timings["geometry"] = round(time.perf_counter() - t0, 3)

        # ── Step 2b: Material sanity (Fix 3) — surfaced as a WARNING, never a cap ──
        material = inspect_material(request.glb_path)
        albedo_rgb = get_albedo_rgb(request.glb_path)

        # ── Step 3: Texture validation + presence gating (alignment-gated) ──
        t0 = time.perf_counter()
        texture_result = validate_texture(
            source_image_path=request.image_path,
            rendered_image_path=aligned_render_path,
            glb_path=request.glb_path,
            alignment=pose,
            albedo_rgb=albedo_rgb,
        )
        presence = check_texture_presence(request.glb_path)
        gate_t = apply_texture_gates(texture_result["score"], presence)
        texture_score = gate_t["score"]
        texture_conf = texture_result.get("confidence", 1.0)
        texture_result["details"]["presence_checks"] = presence
        texture_result["details"]["gating"] = gate_t
        timings["texture"] = round(time.perf_counter() - t0, 3)

        # ── Step 4: Color validation (albedo-preferred, ΔE2000, foreground) ──
        t0 = time.perf_counter()
        color_source = request.image_path  # bg-removed source for clean foreground
        color_result = validate_color(
            source_image_path=color_source,
            rendered_image_path=aligned_render_path,
            albedo_rgb=albedo_rgb,
        )
        # Dominant-palette match uses the albedo texture when available so the
        # color verdict is independent of the renderer's lighting.
        dominant = analyze_dominant_colors(
            color_source, aligned_render_path, render_rgb=albedo_rgb
        )
        dom_score = dominant_color_score(dominant["dominant_color_distance"])
        color_score = round(0.6 * color_result["score"] + 0.4 * dom_score, 1)
        color_result["details"]["dominant_color"] = dominant
        color_result["details"]["dominant_color_score"] = dom_score
        color_result["details"]["material"] = material
        # Color confidence: high when judged on albedo (lighting-independent);
        # otherwise it depends on how well the lit render aligned.
        using_albedo = albedo_rgb is not None and albedo_rgb.size > 0
        color_conf = 0.95 if using_albedo else (geometry_conf if align_trusted else round(geometry_conf * 0.6, 3))
        timings["color"] = round(time.perf_counter() - t0, 3)

        # ── Step 5: Aggregate scores ──
        scores = {
            "geometry": geometry_score,
            "texture": texture_score,
            "color": color_score,
        }
        details = {
            "geometry": geometry_result["details"],
            "texture": texture_result["details"],
            "color": color_result["details"],
        }

        confidences = {
            "geometry": geometry_conf,
            "texture": texture_conf,
            "color": color_conf,
            "alignment_iou": round(float(align_iou), 4) if align_iou is not None else None,
            "alignment_trusted": bool(align_trusted),
        }

        # ── Step 6: Reasoning ──
        t0 = time.perf_counter()
        reasons = generate_reasons(
            scores, details, alignment=pose,
            confidences=confidences, material=material,
        )
        timings["reasoning"] = round(time.perf_counter() - t0, 3)

        # ── Step 7: Evidence overlay ──
        t0 = time.perf_counter()
        try:
            overlay = generate_overlay(request.image_path, aligned_render_path, OUTPUT_DIR)
        except Exception:
            overlay = {"overlay_path": None, "overlay_url": None}
        timings["evidence"] = round(time.perf_counter() - t0, 3)

        timings["total"] = round(time.perf_counter() - t_start, 3)

        return {
            "status": "success",
            "data": {
                "rendered_image": aligned_render_path,
                "rendered_image_url": pose["aligned_render_url"],
                "alignment": {
                    "azimuth": pose["azimuth"],
                    "elevation": pose["elevation"],
                    "iou": pose["iou"],
                    "contour_overlap": pose["contour_overlap"],
                    "confidence": pose["confidence"],
                    "candidates_evaluated": pose["candidates_evaluated"],
                    "input_mask_url": pose["input_mask_url"],
                    "fallback": pose.get("fallback", False),
                },
                "evidence": {
                    "source_url": _to_url(request.image_path),
                    "aligned_render_url": pose["aligned_render_url"],
                    "overlay_url": overlay["overlay_url"],
                },
                "performance": timings,
                "asset_class": asset_class,
                "material_warnings": material.get("warnings", []),
                "confidence": confidences,
                "geometry": {
                    "score": geometry_score,
                    "confidence": geometry_conf,
                    "details": geometry_result["details"],
                    "reason": reasons.get("geometry_reason", ""),
                },
                "texture": {
                    "score": texture_score,
                    "confidence": texture_conf,
                    "details": texture_result["details"],
                    "reason": reasons.get("texture_reason", ""),
                },
                "color": {
                    "score": color_score,
                    "confidence": color_conf,
                    "details": color_result["details"],
                    "reason": reasons.get("color_reason", ""),
                },
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


def _to_url(path: str) -> Optional[str]:
    """Map an on-disk uploads/outputs path to its static URL, if possible."""
    if not path:
        return None
    norm = path.replace("\\", "/")
    for marker in ("/outputs/", "/uploads/"):
        if marker in norm:
            return marker + norm.split(marker, 1)[1]
    return None
