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
from services.geometry_quality_checker import check_geometry_quality, apply_geometry_gates
from services.texture_validator import validate_texture
from services.texture_checker import check_texture_presence, apply_texture_gates
from services.color_validator import validate_color
from services.dominant_color import analyze_dominant_colors, dominant_color_score
from services.evidence_generator import generate_overlay
from services.reason_generator import generate_reasons

router = APIRouter()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ValidateRequest(BaseModel):
    image_path: str                              # bg-removed image
    glb_path: str                                # GLB model
    original_image_path: Optional[str] = None    # original image (optional)


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
        timings["alignment"] = round(time.perf_counter() - t0, 3)

        # ── Step 2: Geometry validation + structural gating ──
        t0 = time.perf_counter()
        geometry_result = validate_geometry(
            source_image_path=request.image_path,
            glb_path=request.glb_path,
            rendered_image_path=aligned_render_path,
        )
        quality = check_geometry_quality(request.glb_path)
        gate_g = apply_geometry_gates(geometry_result["score"], quality)
        geometry_score = gate_g["score"]
        geometry_result["details"]["quality_checks"] = quality
        geometry_result["details"]["gating"] = gate_g
        timings["geometry"] = round(time.perf_counter() - t0, 3)

        # ── Step 3: Texture validation + presence gating ──
        t0 = time.perf_counter()
        texture_result = validate_texture(
            source_image_path=request.image_path,
            rendered_image_path=aligned_render_path,
            glb_path=request.glb_path,
        )
        presence = check_texture_presence(request.glb_path)
        gate_t = apply_texture_gates(texture_result["score"], presence)
        texture_score = gate_t["score"]
        texture_result["details"]["presence_checks"] = presence
        texture_result["details"]["gating"] = gate_t
        timings["texture"] = round(time.perf_counter() - t0, 3)

        # ── Step 4: Color validation + dominant-color analysis ──
        t0 = time.perf_counter()
        color_source = request.image_path  # bg-removed source for clean foreground
        color_result = validate_color(
            source_image_path=color_source,
            rendered_image_path=aligned_render_path,
        )
        dominant = analyze_dominant_colors(color_source, aligned_render_path)
        dom_score = dominant_color_score(dominant["dominant_color_distance"])
        # Blend distribution-level color (existing) with perceptual dominant shift.
        color_score = round(0.6 * color_result["score"] + 0.4 * dom_score, 1)
        color_result["details"]["dominant_color"] = dominant
        color_result["details"]["dominant_color_score"] = dom_score
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

        # ── Step 6: Reasoning ──
        t0 = time.perf_counter()
        reasons = generate_reasons(scores, details, alignment=pose)
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
                "geometry": {
                    "score": geometry_score,
                    "details": geometry_result["details"],
                    "reason": reasons.get("geometry_reason", ""),
                },
                "texture": {
                    "score": texture_score,
                    "details": texture_result["details"],
                    "reason": reasons.get("texture_reason", ""),
                },
                "color": {
                    "score": color_score,
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
