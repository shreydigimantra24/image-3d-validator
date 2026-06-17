# Image-to-3D Object Validation Pipeline

An end-to-end system that validates 3D GLB models against source product images, producing quantitative Geometry, Texture, and Color scores with human-readable explanations.

## Features

- **Background Removal** — Isolate products using RMBG-2.0 (BriaAI)
- **3D Model Upload** — Upload GLB/glTF files with interactive 3D preview
- **GLB Rendering** — Render 3D models to 2D images via PyRender/Trimesh
- **Geometry Validation** — Mesh integrity + silhouette matching (IoU, Chamfer, Hausdorff)
- **Texture Validation** — Texture presence + SSIM + LPIPS perceptual comparison
- **Color Validation** — LAB Delta E + histogram correlation + Earth Mover's Distance
- **LLM Reasoning** — Human-readable explanations via Groq (Llama 3.3)

## Architecture

```
User Upload → Background Removal (RMBG-2.0) → Preprocessed Image
                                                    │
                                    ┌───────────────┤
                                    ▼               ▼
                              Upload GLB      (Future: Generate GLB)
                                    │
                                    ▼
                           ┌──────────────────┐
                           │ Validation Engine │
                           ├──────────────────┤
                           │ Geometry (40/60)  │
                           │ Texture (20/40/40)│
                           │ Color   (70/30)   │
                           └────────┬─────────┘
                                    ▼
                           Scores & LLM Reasons
```

## Tech Stack

| Layer           | Technology                        |
|-----------------|-----------------------------------|
| Backend         | FastAPI, Uvicorn                  |
| Frontend        | React (Vite), Google Model Viewer |
| Background Rem. | RMBG-2.0, HuggingFace Transformers|
| Mesh Processing | Trimesh, PyRender, Open3D         |
| Image Metrics   | OpenCV, scikit-image, LPIPS       |
| Color Metrics   | NumPy, SciPy, OpenCV (LAB)        |
| LLM             | Groq API (Llama 3.3 70B)          |

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- GROQ_API_KEY (optional, for LLM reasoning)

### Backend Setup

```bash
cd image-3d-validator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set Groq API key (optional)
export GROQ_API_KEY="your-api-key-here"

# Start backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd image-3d-validator/frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Open `http://localhost:3000` in your browser.

## Validation Methodology

### Geometry Score (0-100)

```
geometry_score = 0.4 × mesh_integrity + 0.6 × silhouette_similarity
```

- **Mesh Integrity**: Watertightness, manifold check, degenerate faces, floating components
- **Silhouette Matching**: IoU, Chamfer Distance, Hausdorff Distance

### Texture Score (0-100)

```
texture_score = 0.2 × texture_presence + 0.4 × SSIM + 0.4 × LPIPS
```

- **Texture Presence**: Material, texture file, UV coordinates
- **SSIM**: Structural similarity (0→1)
- **LPIPS**: Perceptual similarity (lower = better)

### Color Score (0-100)

```
color_score = 0.7 × deltaE_score + 0.3 × histogram_similarity
```

- **Delta E (CIE76)**: Per-pixel LAB color difference
- **Histogram Similarity**: Correlation + Earth Mover's Distance per LAB channel

## Project Structure

```
image-3d-validator/
├── backend/
│   ├── api/
│   │   ├── upload.py          # Image & GLB upload endpoints
│   │   ├── preprocess.py      # Background removal endpoint
│   │   └── validate.py        # Validation orchestration endpoint
│   ├── services/
│   │   ├── background_removal.py   # RMBG-2.0 integration
│   │   ├── glb_renderer.py         # PyRender/Trimesh rendering
│   │   ├── geometry_validator.py    # Mesh integrity + silhouette
│   │   ├── texture_validator.py     # SSIM + LPIPS + presence
│   │   ├── color_validator.py       # Delta E + histograms
│   │   └── reason_generator.py      # Groq LLM reasoning
│   └── main.py
├── frontend/
│   └── src/
│       ├── App.jsx            # Main React component
│       ├── App.css            # Component styles
│       ├── index.css          # Design system
│       └── main.jsx           # Entry point
├── uploads/                   # Uploaded files
├── outputs/                   # Processed outputs
├── requirements.txt
└── README.md
```

## Limitations

- GLB rendering depends on OpenGL availability (headless fallback provided)
- LPIPS requires PyTorch; falls back to MSE-based proxy if unavailable
- Background removal model downloads ~500MB on first run
- Camera angle matching between source image and GLB render is approximate

## Future Improvements

- GLB generation (TripoSR, Hunyuan3D, InstantMesh)
- Multi-view validation
- Batch processing
- Export validation reports as PDF
- Fine-tuned camera angle estimation
