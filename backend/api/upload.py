"""
Upload API — handles image and GLB file uploads.
"""

import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_MODEL_EXTENSIONS = {".glb", ".gltf"}


def _save_upload(file: UploadFile, allowed_exts: set, subfolder: str) -> dict:
    """Save an uploaded file to disk and return metadata."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_exts}",
        )

    file_id = str(uuid.uuid4())
    dest_dir = os.path.join(UPLOAD_DIR, subfolder)
    os.makedirs(dest_dir, exist_ok=True)

    filename = f"{file_id}{ext}"
    filepath = os.path.join(dest_dir, filename)

    with open(filepath, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    return {
        "file_id": file_id,
        "filename": filename,
        "filepath": filepath,
        "url": f"/uploads/{subfolder}/{filename}",
    }


@router.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload a product image."""
    result = _save_upload(file, ALLOWED_IMAGE_EXTENSIONS, "images")
    return {"status": "success", "data": result}


@router.post("/upload/glb")
async def upload_glb(file: UploadFile = File(...)):
    """Upload a GLB/glTF 3D model."""
    result = _save_upload(file, ALLOWED_MODEL_EXTENSIONS, "models")
    return {"status": "success", "data": result}
