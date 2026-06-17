# Image-to-3D Object Validation Pipeline

An end-to-end system that validates 3D GLB models against source product images, producing quantitative Geometry, Texture, and Color scores with human-readable explanations.

## Features

- **Background Removal** — Isolate products using RMBG-2.0 (BriaAI)
- **3D Model Upload** — Upload GLB/glTF files with interactive 3D preview
- **GLB Rendering** — Render 3D models to 2D images via PyRender/Trimesh
- **Geometry Validation** — Mesh integrity + silhouette matching (IoU, Chamfer, Hausdorff)
- **Texture Validation** — Texture presence + SSIM + LPIPS perceptual comparison
- **Color Validation** — LAB Delta E + histogram correlation + Earth Mover's Distance
- **Camera Pose Alignment** — Silhouette-search the viewpoint that matches the input before scoring
- **Validation Evidence** — Source / aligned render / overlay panel for visual proof
- **Geometry Quality Gating** — Holes, non-manifold edges, degenerate faces, components, normals cap the score
- **Texture Presence Gating** — Missing material / UV / texture image heavily penalized before similarity is trusted
- **Dominant Color Analysis** — KMeans palette matching catches perceptual color shifts (e.g. red → orange)
- **Performance Monitoring** — Per-stage latency breakdown
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

### 0. Camera Pose Alignment (runs first)

Before any scoring, the validator extracts a binary silhouette from the
background-removed image, renders the GLB across an azimuth × elevation grid
(0–330° in 30° steps × {-15, 0, 15, 30}°), scores each candidate by
`0.7 × IoU + 0.3 × contour_overlap`, and keeps the best-matching **aligned
render**. All geometry, texture, and color scores are computed from this
viewpoint so the image and model are compared from the same angle.

### Geometry Score (0-100)

```
base   = 0.4 × mesh_integrity + 0.6 × silhouette_similarity
score  = apply_geometry_gates(base, quality_checks)
```

- **Silhouette Matching**: IoU, Chamfer Distance, Hausdorff Distance (on the aligned render)
- **Quality Checks**: holes, non-manifold edges, degenerate faces, floating components, normal consistency
- **Gating**: e.g. `holes > 10 → cap 40`, `components > 5 → cap 50`, inconsistent normals → cap 70

### Texture Score (0-100)

```
base   = 0.2 × texture_presence + 0.4 × SSIM + 0.4 × LPIPS
score  = apply_texture_gates(base, presence_checks)
```

- **Presence Checks**: material, UV coordinates, base-color texture image
- **Gating**: no material → cap 10, no texture image → cap 25, no UVs → cap 30
- **SSIM / LPIPS**: structural + perceptual similarity (LPIPS falls back to MSE proxy)

### Color Score (0-100)

```
distribution = 0.7 × deltaE_score + 0.3 × histogram_similarity
dominant     = dominant_color_score(KMeans palette ΔE)
score        = 0.6 × distribution + 0.4 × dominant
```

- **Delta E (CIE76)**: per-pixel LAB color difference
- **Histogram Similarity**: correlation + Earth Mover's Distance per LAB channel
- **Dominant Color**: KMeans(5) foreground palettes matched in LAB; flags primary-hue shifts

### Performance Monitoring

Each stage (`alignment`, `geometry`, `texture`, `color`, `reasoning`,
`evidence`) is timed and returned under `data.performance`, with a `total`.

### Calibration Suite

`backend/tests/degradation_tests.py` proves metric responsiveness by degrading
a copy of the asset and re-scoring:

```bash
cd backend
GLB_PATH=/path/model.glb IMAGE_PATH=/path/source.png python -m tests.degradation_tests
```

| Test Case        | Geometry | Texture | Color |
| ---------------- | -------- | ------- | ----- |
| Original         | high     | high    | high  |
| Missing Faces    | ↓        | stable  | stable|
| Missing Texture  | stable   | ↓       | stable|
| Hue Shift        | stable   | stable  | ↓     |

## Project Structure

```
image-3d-validator/
├── backend/
│   ├── api/
│   │   ├── upload.py          # Image & GLB upload endpoints
│   │   ├── preprocess.py      # Background removal endpoint
│   │   └── validate.py        # Validation orchestration endpoint
│   ├── services/
│   │   ├── background_removal.py        # RMBG-2.0 integration
│   │   ├── glb_renderer.py              # Pose-aware PyRender/Trimesh rendering
│   │   ├── pose_estimator.py            # Camera pose search (silhouette IoU)
│   │   ├── geometry_validator.py        # Mesh integrity + silhouette
│   │   ├── geometry_quality_checker.py  # Holes/manifold/degenerate + score gating
│   │   ├── texture_validator.py         # SSIM + LPIPS + presence
│   │   ├── texture_checker.py           # Material/UV/texture presence + gating
│   │   ├── color_validator.py           # Delta E + histograms
│   │   ├── dominant_color.py            # KMeans dominant-color analysis
│   │   ├── evidence_generator.py        # Overlay (source + aligned render)
│   │   └── reason_generator.py          # Groq LLM reasoning
│   ├── tests/
│   │   └── degradation_tests.py         # Calibration / degradation suite
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
