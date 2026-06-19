# Image-to-3D Object Validation Pipeline

An end-to-end system that validates 3D GLB models against source product images, producing quantitative Geometry, Texture, and Color scores with human-readable explanations.

## Live Demo

**Live URL:** https://image-3d-validator.dmlabs.in/

### How to add examples and test on the live URL

The repo ships ready-to-use example pairs in [`examples/`](examples/):

| Image                  | Model                  |
|------------------------|------------------------|
| `examples/image1.png`  | `examples/model1.glb`  |
| `examples/image2.png`  | `examples/model2.glb`  |

1. Open https://image-3d-validator.dmlabs.in/ in your browser.
2. **Upload the source image** — pick one of the example PNGs (e.g. `examples/image1.png`). Background removal (RMBG-2.0) runs automatically.
3. **Upload the matching GLB** — pick the paired model (e.g. `examples/model1.glb`). Use the interactive 3D preview to confirm it loaded.
4. Click **Validate**. The pipeline runs pose alignment → geometry → texture → color and returns scores with LLM reasons.
5. Review the **Geometry / Texture / Color** scores, the validation-evidence panel (source / aligned render / overlay), and the per-stage performance breakdown.

> Always pair the matching image + model (`image1`↔`model1`, `image2`↔`model2`) so the silhouette alignment is meaningful. To test mismatch behavior, deliberately cross a pair (e.g. `image1.png` + `model2.glb`) and watch the scores drop.

## Features

- **Background Removal** — Isolate products using RMBG-2.0 (BriaAI)
- **3D Model Upload** — Upload GLB/glTF files with interactive 3D preview
- **GLB Rendering** — Render 3D models to 2D images via PyRender/Trimesh
- **Asset-Class-Aware Geometry** — Product/assembly meshes (many open-shell parts) are scored render-first and NOT penalized for non-watertightness / component / hole counts; single solids keep legacy topology gates
- **Geometry Validation** — Silhouette matching (IoU, Chamfer, Hausdorff) + genuine-defect detection (NaN/inf, normals, degenerate faces, floaters/slivers)
- **Texture Validation** — Texture presence + SSIM + LPIPS, gated on alignment confidence (robust LAB-histogram fallback when IoU is low)
- **Color Validation** — Albedo-first, foreground-masked LAB **ΔE2000** + histogram; luminance-normalized render fallback
- **Material Sanity Check** — Flags suspicious PBR (e.g. `metallicFactor > 0.8` on a textured asset) as a warning, separate from the color score
- **Camera Pose Alignment** — Finer-grid silhouette search + local/joint refinement; alignment confidence propagated into every score
- **Validation Evidence** — Source / aligned render / overlay panel for visual proof
- **Dominant Color Analysis** — KMeans palette matching in ΔE2000 catches perceptual color shifts
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
                           │ Geometry (asset-aware, render-first) │
                           │ Texture  (alignment-gated SSIM/LPIPS)│
                           │ Color    (albedo ΔE2000, masked)     │
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
| Mesh Processing | Trimesh, PyRender, PyOpenGL       |
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
# Login using Hugging face auth login
hf auth login

cd image-3d-validator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set Groq API key (optional) — copy the template and edit
cp .env.example .env
# then put your key in .env:  GROQ_API_KEY=your-api-key-here

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

### Asset class (product vs single solid)

Scoring is **asset-class aware**. Most catalog assets are *products / assemblies*
(furniture, appliances): by design they are made of **many separate open-shell
parts** — a table + 4 chairs is thousands of disconnected components, is **not
watertight**, and has tens of thousands of open boundary edges. None of those are
defects for this class, so they must not lower the geometry score.

- `asset_class` is taken from the request, or **auto-detected**: a mesh with
  many connected components **and** several *substantial* parts (≥ 200 faces) is
  treated as a `product`. A single watertight solid (scan / printable part) is a
  `single_solid`, where the legacy topology gates still apply.
- Configurable via `validation_config.py` / env (`ASSET_CLASS`,
  `SUBSTANTIAL_PART_MIN_FACES`, …).

### Geometry Score (0-100) — render-first, asset-aware

```
product:       score = structural_soundness × silhouette_factor(IoU)
single_solid:  score = 0.4 × mesh_integrity + 0.6 × silhouette   (legacy)
score = apply_geometry_gates(score, quality, asset_class)
```

- **Render-derived primary signal**: silhouette IoU/Chamfer/Hausdorff from the
  aligned view. `silhouette_factor` maps IoU → [0.6, 1.0] so a structurally
  sound mesh stays high unless the rendered **shape** genuinely disagrees.
- **Structural soundness (genuine defects only)**: NaN/inf vertices, inconsistent
  normals, degenerate (zero-area) faces, isolated 1–2 face slivers far from the
  body, and substantial components flung far outside the main bbox (true
  floaters/spikes).
- **NOT penalized for products**: non-watertightness, connected-component count,
  open boundary edges. These are reported as descriptors only — they never cap
  a product score, and the reason text never cites them as defects.

### Texture Score (0-100) — alignment-gated

```
trusted (IoU ≥ threshold):  base = 0.2 × presence + 0.4 × SSIM + 0.4 × LPIPS
low IoU (fallback):         base = 0.5 × presence + 0.5 × foreground_LAB_histogram
score = apply_texture_gates(base, presence_checks)
```

- **Alignment-confidence gating**: SSIM/LPIPS compare per-pixel content, which is
  only meaningful when the silhouettes overlap tightly. Below
  `IOU_TRUST_THRESHOLD` (default **0.85**, configurable) those metrics are
  down-weighted in favour of an alignment-robust foreground LAB histogram, and a
  **reduced `confidence`** is reported on the score and in the reason text.
- **Presence/Gating**: material, UVs, base-color texture image — no material →
  cap 10, no texture image → cap 25, no UVs → cap 30.

### Color Score (0-100) — albedo-first, lighting-normalized, ΔE2000

```
distribution = 0.7 × deltaE2000_score + 0.3 × histogram_similarity   (foreground-masked)
dominant     = dominant_color_score(KMeans palette ΔE2000)
score        = 0.6 × distribution + 0.4 × dominant
```

- **Albedo, not the lit render**: the model's color reference is the asset's
  **baseColor (albedo) texture**, modulated by `baseColorFactor` — independent
  of how the renderer lit it. If no albedo texture exists we fall back to the
  render with its **luminance normalized** to the source (cancels exposure /
  the metallic-dark effect without erasing genuine hue shifts).
- **Metallic-material caveat (warning, not a penalty)**: a textured fabric/painted
  asset reported as `metallicFactor > 0.8` renders dark **without an environment
  map / IBL** even though its albedo is correct. This is surfaced as a separate
  **material warning** ("albedo matches; metallic=1.0 may cause a dark appearance
  under lighting"); it does **not** tank the color score.
- **ΔE2000 (CIEDE2000)**: modern perceptual color difference (replaces CIE76),
  computed **foreground-masked** in CIE Lab.
- **Lighting / IBL**: renders use a neutral ambient fill (IBL stand-in) + fixed
  key/fill so materials show near their true albedo.

### Camera pose & alignment confidence

The coarse pose scan uses a **finer grid** (20° azimuth × 6 elevations, then a
two-scale local silhouette refinement and a bounded joint optimizer that
maximizes silhouette IoU). The resulting IoU drives the **confidence** propagated
into every score and reason string, and the texture-metric trust gate above.

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

### Reference-Case Regression Test

`backend/tests/reference_case_test.py` locks in the corrected behaviour on the
known failing case — a multi-part product/assembly with a **light albedo** but
**`metallicFactor = 1.0`** (previously scored Geometry 40 / Texture 76 / Color 60
with charcoal chairs). It asserts: asset auto-classified `product`; **geometry ≥ 80**
(structurally fine, no watertight/component/hole gate); a **material warning** for
metallic; **color not falsely low** (judged on albedo); and **confidence reported**.

It runs with no proprietary asset (synthetic fixture, open-source deps only), or
against the real GLB:

```bash
cd backend
# synthetic fixture (default):
python -m tests.reference_case_test
# real asset:
REF_GLB_PATH=/path/vihals.glb REF_IMAGE_PATH=/path/source.png python -m tests.reference_case_test
```

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
│   │   ├── geometry_validator.py        # Render-first, asset-aware geometry score
│   │   ├── geometry_quality_checker.py  # Genuine-defect detection + asset-aware gating
│   │   ├── texture_validator.py         # SSIM/LPIPS (alignment-gated) + presence
│   │   ├── texture_checker.py           # Material/UV/texture presence + gating
│   │   ├── color_validator.py           # Albedo-first ΔE2000 (foreground-masked)
│   │   ├── dominant_color.py            # KMeans palette analysis (ΔE2000)
│   │   ├── material_inspector.py        # PBR/metallic inspection + albedo extraction
│   │   ├── validation_config.py         # Tunables (asset class, IoU trust, metallic)
│   │   ├── evidence_generator.py        # Overlay (source + aligned render)
│   │   └── reason_generator.py          # Groq LLM reasoning (confidence + material aware)
│   ├── tests/
│   │   ├── degradation_tests.py         # Calibration / degradation suite
│   │   └── reference_case_test.py       # Reference-case regression (Fixes 1-4)
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

# Current Limitations & Future Improvements

## 1. Pose Alignment Accuracy (~64% Silhouette IoU)

The current alignment process relies on a global azimuth/elevation search with limited local refinement. While effective for simple objects, it can struggle with complex scenes, articulated structures, or assets whose spatial arrangement differs from the reference image. Alignment quality is currently the primary bottleneck for downstream texture and color validation confidence.

**Planned Improvements**

* Finer pose sampling and search resolution
* Local optimization (e.g., Nelder–Mead)
* Differentiable-render refinement
* Optional per-object alignment for multi-part scenes

---

## 2. Alignment-Dependent Texture Validation

Texture metrics such as SSIM and LPIPS require strong pixel-level correspondence between the rendered model and the reference image. When silhouette overlap falls below the confidence threshold, these metrics are automatically down-weighted and replaced with more alignment-robust histogram-based comparisons. In such cases, lower confidence is reported rather than incorrectly flagging texture defects.

**Planned Improvements**

* Improved pose alignment to increase per-pixel metric reliability
* Adaptive confidence estimation based on alignment quality

---

## 3. Single-View Validation

Validation is currently performed against a single reference image. As a result, only geometry, texture, and color visible from that viewpoint can be evaluated. Occluded regions, back-facing surfaces, and unseen portions of the asset remain unassessed.

**Planned Improvements**

* Multi-view validation support
* Novel-view consistency checks
* Aggregated scoring across multiple viewpoints

---

## 4. Rendering & Lighting Sensitivity

Texture and color evaluation can be influenced by rendering conditions. Materials such as metals, glass, or highly reflective surfaces may appear significantly different depending on lighting and environment settings. While direct albedo-texture analysis reduces this dependency, full appearance fidelity is not yet measured.

**Planned Improvements**

* Physically Based Rendering (PBR) under matched environment maps
* Exposure and white-balance normalization
* Lighting-invariant appearance comparison

---

## 5. Inference Performance (~25s per Asset)

Validation runtime is currently dominated by pose alignment (~8–9 seconds), which relies on a brute-force search across candidate viewpoints.

**Planned Improvements**

* Coarse-to-fine pose search
* Low-resolution candidate filtering followed by high-resolution refinement
* GPU batching
* Model and renderer caching

---

## 6. Score Calibration

Quality scores are currently derived from hand-tuned metric thresholds rather than a benchmark dataset of human-rated examples. Consequently, score interpretation may not always align perfectly with human perception.

**Planned Improvements**

* Calibration using labeled good/bad asset pairs
* Learned metric-to-score mappings
* Continuous score validation against human feedback

---

## 7. Heuristic Asset Classification

The pipeline infers whether an asset is a single object or a multi-component assembly using structural heuristics. Unusual asset structures may occasionally be misclassified, affecting geometry evaluation behavior.

**Planned Improvements**

* Dedicated asset-type classification models
* User-configurable validation modes
* Asset-specific scoring strategies

---

## 8. Limited Artifact Detection Coverage

Current geometry validation focuses primarily on structural defects, mesh quality, and silhouette consistency. More subtle issues such as UV seams, texture stretching, baking artifacts, and shading inconsistencies are not comprehensively detected.

**Planned Improvements**

* UV-space analysis
* Texture distortion metrics
* Render-space artifact detection
* Baking quality assessment

---

## 9. Global Color Evaluation

Color validation currently relies on overall foreground appearance and dominant color distributions. Localized color errors affecting only a small region may be diluted within the global score.

**Planned Improvements**

* Region-based color comparison
* Semantic-part segmentation
* Per-component color consistency checks

---

## 10. Edge Case Robustness

Special cases such as untextured assets, highly symmetric objects, transparent materials, reflective surfaces, and corrupted meshes are not yet fully hardened across all validation stages.

**Planned Improvements**

* Dedicated handling logic for challenging asset categories
* Expanded regression test suite
* Improved failure detection and reporting
