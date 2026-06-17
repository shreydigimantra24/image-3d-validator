"""
Validate API — orchestrates geometry, texture, and color validation.
"""

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from services.glb_renderer import render_glb
from services.geometry_validator import validate_geometry
from services.texture_validator import validate_texture
from services.color_validator import validate_color
from services.reason_generator import generate_reasons

router = APIRouter()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ValidateRequest(BaseModel):
    image_path: str        # Path to the bg-removed image
    glb_path: str          # Path to the GLB model
    original_image_path: Optional[str] = None  # Path to original image (for color)


@router.post("/validate")
async def validate_model(request: ValidateRequest):
    """
    Full validation pipeline:
    1. Render the GLB to a 2D image
    2. Run geometry validation
    3. Run texture validation
    4. Run color validation
    5. Generate LLM-based reasoning
    """
    # Verify files exist
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Preprocessed image not found")
    if not os.path.exists(request.glb_path):
        raise HTTPException(status_code=404, detail="GLB model not found")

    try:
        # Step 1: Render GLB
        rendered_path = render_glb(request.glb_path, OUTPUT_DIR)

        # Step 2: Geometry Validation
        geometry_result = validate_geometry(
            source_image_path=request.image_path,
            glb_path=request.glb_path,
            rendered_image_path=rendered_path,
        )

        # Step 3: Texture Validation
        texture_result = validate_texture(
            source_image_path=request.image_path,
            rendered_image_path=rendered_path,
            glb_path=request.glb_path,
        )

        # Step 4: Color Validation
        color_image = request.original_image_path or request.image_path
        color_result = validate_color(
            source_image_path=color_image,
            rendered_image_path=rendered_path,
        )

        # Step 5: Generate reasons
        scores = {
            "geometry": geometry_result["score"],
            "texture": texture_result["score"],
            "color": color_result["score"],
        }
        details = {
            "geometry": geometry_result["details"],
            "texture": texture_result["details"],
            "color": color_result["details"],
        }
        reasons = generate_reasons(scores, details)

        return {
            "status": "success",
            "data": {
                "rendered_image": rendered_path,
                "rendered_image_url": f"/outputs/{os.path.basename(rendered_path)}",
                "geometry": {
                    "score": geometry_result["score"],
                    "details": geometry_result["details"],
                    "reason": reasons.get("geometry_reason", ""),
                },
                "texture": {
                    "score": texture_result["score"],
                    "details": texture_result["details"],
                    "reason": reasons.get("texture_reason", ""),
                },
                "color": {
                    "score": color_result["score"],
                    "details": color_result["details"],
                    "reason": reasons.get("color_reason", ""),
                },
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
