"""
Image-to-3D Object Validation Pipeline - FastAPI Backend
"""

import os
from dotenv import load_dotenv, find_dotenv

# Load .env before importing routers/services — several modules read env vars
# (GROQ_API_KEY, ASSET_CLASS, IoU thresholds) at import time.
load_dotenv(find_dotenv(usecwd=True))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.upload import router as upload_router
from api.preprocess import router as preprocess_router
from api.validate import router as validate_router

app = FastAPI(
    title="Image-to-3D Validator",
    description="Validate 3D GLB models against source product images",
    version="1.0.0",
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload/output directories exist
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve uploaded and output files as static
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Register API routes
app.include_router(upload_router, prefix="/api", tags=["Upload"])
app.include_router(preprocess_router, prefix="/api", tags=["Preprocess"])
app.include_router(validate_router, prefix="/api", tags=["Validate"])


@app.get("/")
async def root():
    return {"message": "Image-to-3D Validator API is running"}
