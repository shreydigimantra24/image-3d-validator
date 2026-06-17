"""
Preprocess API — handles background removal.
"""

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.background_removal import remove_background

router = APIRouter()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class PreprocessRequest(BaseModel):
    image_path: str  # Path to the uploaded image


@router.post("/preprocess")
async def preprocess_image(request: PreprocessRequest):
    """Remove background from the uploaded product image."""
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image file not found")

    try:
        result = remove_background(request.image_path, OUTPUT_DIR)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")
