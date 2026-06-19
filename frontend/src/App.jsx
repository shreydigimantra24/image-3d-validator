import { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

// Backend origin. Empty by default → relative paths (works with the Vite dev
// proxy or when frontend + backend share an origin behind one reverse proxy).
// For a separately-hosted backend, set VITE_BACKEND_URL at build time, e.g.
//   VITE_BACKEND_URL=https://api.example.com
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || '';
const API_BASE = `${BACKEND_URL}/api`;
// Prefix backend-served asset paths (/uploads/.., /outputs/..) with the origin.
const asset = (path) => (path ? `${BACKEND_URL}${path}` : path);

function ScoreRing({ score, label }) {
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;

  const scoreClass = score >= 75 ? 'score-high' : score >= 50 ? 'score-mid' : 'score-low';
  const strokeColor =
    score >= 75 ? 'var(--accent-success)' : score >= 50 ? 'var(--accent-warning)' : 'var(--accent-danger)';

  return (
    <div className="score-card">
      <div className="score-card__label">{label}</div>
      <div className="score-ring">
        <svg viewBox="0 0 100 100">
          <circle className="score-ring__bg" cx="50" cy="50" r={radius} />
          <circle
            className="score-ring__fill"
            cx="50"
            cy="50"
            r={radius}
            stroke={strokeColor}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        <div className={`score-ring__text ${scoreClass}`}>{Math.round(score)}</div>
      </div>
    </div>
  );
}

function QualityChips({ quality }) {
  if (!quality) return null;
  const items = [
    ['Holes', quality.holes],
    ['Components', quality.components],
    ['Degenerate', quality.degenerate_faces],
    ['Non-manifold', quality.non_manifold_edges],
    ['Watertight', quality.is_watertight ? 'yes' : 'no'],
    ['Normals', quality.normals_consistent ? 'ok' : 'bad'],
  ];
  return (
    <div className="chip-row">
      {items.map(([label, val]) => (
        <span className="metric-chip" key={label}>
          {label}: <strong>{String(val)}</strong>
        </span>
      ))}
    </div>
  );
}

function PresenceChips({ presence }) {
  if (!presence) return null;
  const items = [
    ['Material', presence.material_present],
    ['Texture', presence.texture_present],
    ['UVs', presence.uv_present],
    ['Vertex colors', presence.has_vertex_colors],
  ];
  return (
    <div className="chip-row">
      {items.map(([label, ok]) => (
        <span className={`metric-chip ${ok ? 'chip-ok' : 'chip-bad'}`} key={label}>
          {label}: <strong>{ok ? 'yes' : 'no'}</strong>
        </span>
      ))}
    </div>
  );
}

function GateChips({ gating }) {
  if (!gating || !gating.gated) return null;
  return (
    <div className="chip-row">
      {gating.applied_gates.map((g, i) => (
        <span className="metric-chip chip-gate" key={i}>{g}</span>
      ))}
    </div>
  );
}

function DominantPalettes({ dominant }) {
  if (!dominant || !dominant.source_palette) return null;
  const swatches = (palette) => (
    <div className="swatch-row">
      {palette.map((c, i) => (
        <span
          key={i}
          className="swatch"
          title={`rgb(${c.rgb.join(',')}) · ${Math.round(c.weight * 100)}%`}
          style={{ background: `rgb(${c.rgb.join(',')})`, flex: Math.max(c.weight, 0.05) }}
        />
      ))}
    </div>
  );
  return (
    <div className="palette-block">
      <div className="palette-row">
        <span className="palette-label">Source</span>
        {swatches(dominant.source_palette)}
      </div>
      <div className="palette-row">
        <span className="palette-label">Render</span>
        {swatches(dominant.render_palette)}
      </div>
      <div className="palette-distance">
        Dominant color distance (ΔE): <strong>{dominant.dominant_color_distance}</strong>
      </div>
    </div>
  );
}

function PerformancePanel({ timings }) {
  const order = ['alignment', 'geometry', 'texture', 'color', 'reasoning', 'evidence'];
  const total = timings.total || order.reduce((s, k) => s + (timings[k] || 0), 0);
  return (
    <div className="perf-panel">
      <div className="reason-card__header">
        <span className="reason-card__title">Performance ({total}s total)</span>
      </div>
      <div className="perf-bars">
        {order.filter((k) => timings[k] != null).map((k) => (
          <div className="perf-row" key={k}>
            <span className="perf-label">{k}</span>
            <div className="perf-track">
              <div
                className="perf-fill"
                style={{ width: `${Math.min(100, (timings[k] / (total || 1)) * 100)}%` }}
              />
            </div>
            <span className="perf-value">{timings[k]}s</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Pipeline phases shown while processing
const PHASES = ['Uploading image', 'Removing background', 'Uploading 3D model', 'Analyzing model'];

function App() {
  // Selected inputs (both required before anything runs)
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  const [glbFile, setGlbFile] = useState(null);
  const [glbUrl, setGlbUrl] = useState(null);

  // Pipeline outputs
  const [bgRemovedUrl, setBgRemovedUrl] = useState(null);
  const [renderedImageUrl, setRenderedImageUrl] = useState(null);
  const [validationResult, setValidationResult] = useState(null);

  // Single combined run state
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState(-1);
  const [status, setStatus] = useState({ type: '', message: '' });

  const imageInputRef = useRef(null);
  const glbInputRef = useRef(null);

  const bothReady = !!imageFile && !!glbFile;

  // ─── Input selection ───
  const pickImage = (file) => {
    if (!file) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    resetOutputs();
  };

  const pickGlb = (file) => {
    if (!file) return;
    setGlbFile(file);
    setGlbUrl(URL.createObjectURL(file));
    resetOutputs();
  };

  const resetOutputs = () => {
    setBgRemovedUrl(null);
    setRenderedImageUrl(null);
    setValidationResult(null);
    setStatus({ type: '', message: '' });
    setPhase(-1);
  };

  const handleImageSelect = (e) => pickImage(e.target.files[0]);
  const handleGlbSelect = (e) => pickGlb(e.target.files[0]);

  const clearImage = () => {
    setImageFile(null);
    setImagePreview(null);
    resetOutputs();
  };
  const clearGlb = () => {
    setGlbFile(null);
    setGlbUrl(null);
    resetOutputs();
  };

  // ─── Drag & Drop ───
  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('active');
  };
  const handleDragLeave = (e) => e.currentTarget.classList.remove('active');
  const handleImageDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('active');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) pickImage(file);
  };
  const handleGlbDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('active');
    const file = e.dataTransfer.files[0];
    if (file && (file.name.endsWith('.glb') || file.name.endsWith('.gltf'))) pickGlb(file);
  };

  // ─── Full pipeline — only runs when both inputs present ───
  const runPipeline = async () => {
    if (!bothReady || running) return;
    setRunning(true);
    setValidationResult(null);
    setBgRemovedUrl(null);
    setRenderedImageUrl(null);

    try {
      // 1. Upload image
      setPhase(0);
      setStatus({ type: 'info', message: PHASES[0] });
      const imgForm = new FormData();
      imgForm.append('file', imageFile);
      const uploadRes = await axios.post(`${API_BASE}/upload/image`, imgForm);
      const uploadedPath = uploadRes.data.data.filepath;

      // 2. Remove background
      setPhase(1);
      setStatus({ type: 'info', message: PHASES[1] });
      const preprocessRes = await axios.post(`${API_BASE}/preprocess`, { image_path: uploadedPath });
      const bgResult = preprocessRes.data.data;
      setBgRemovedUrl(bgResult.output_url);

      // 3. Upload GLB
      setPhase(2);
      setStatus({ type: 'info', message: PHASES[2] });
      const glbForm = new FormData();
      glbForm.append('file', glbFile);
      const glbRes = await axios.post(`${API_BASE}/upload/glb`, glbForm);
      const glbPath = glbRes.data.data.filepath;

      // 4. Validate
      setPhase(3);
      setStatus({ type: 'info', message: PHASES[3] });
      const res = await axios.post(`${API_BASE}/validate`, {
        image_path: bgResult.output_path,
        glb_path: glbPath,
        original_image_path: uploadedPath,
      });
      const data = res.data.data;
      setValidationResult(data);
      setRenderedImageUrl(data.rendered_image_url);
      setStatus({ type: 'success', message: 'Validation complete' });
      setPhase(-1);
    } catch (err) {
      console.error(err);
      setStatus({ type: 'error', message: `Failed: ${err.response?.data?.detail || err.message}` });
      setPhase(-1);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Image to 3D Validator</h1>
        <p>Compare a 3D model against its source product image with precision scoring</p>
      </header>

      <div className="steps">
        {/* ───────── Inputs — both required ───────── */}
        <section className="step-section">
          <div className="step-header">
            <div className="step-number">1</div>
            <div>
              <div className="step-title">Add your files</div>
              <div className="step-subtitle">A product image and a 3D model are both required to begin</div>
            </div>
          </div>

          <div className="input-grid">
            {/* Image input */}
            <div className="input-col">
              <div className="input-col__title">Product image</div>
              {!imagePreview ? (
                <div
                  className="upload-zone"
                  onClick={() => imageInputRef.current?.click()}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleImageDrop}
                >
                  <input ref={imageInputRef} type="file" accept="image/*" onChange={handleImageSelect} />
                  <div className="upload-zone__text">
                    <strong>Click to upload</strong> or drag &amp; drop
                  </div>
                  <div className="upload-zone__hint">JPG, PNG or WebP, up to 50MB</div>
                </div>
              ) : (
                <div className="preview-frame">
                  <div className="image-preview">
                    <div className="image-preview__label">Selected image</div>
                    <img src={imagePreview} alt="Selected product" />
                  </div>
                  <button className="btn btn-ghost btn-sm" onClick={clearImage} disabled={running}>
                    Replace
                  </button>
                </div>
              )}
            </div>

            {/* GLB input */}
            <div className="input-col">
              <div className="input-col__title">3D model</div>
              {!glbUrl ? (
                <div
                  className="upload-zone"
                  onClick={() => glbInputRef.current?.click()}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleGlbDrop}
                >
                  <input ref={glbInputRef} type="file" accept=".glb,.gltf" onChange={handleGlbSelect} />
                  <div className="upload-zone__text">
                    <strong>Click to upload</strong> or drag &amp; drop
                  </div>
                  <div className="upload-zone__hint">GLB or glTF format</div>
                </div>
              ) : (
                <div className="preview-frame">
                  <div className="viewer-container">
                    <model-viewer
                      src={glbUrl}
                      alt="3D model preview"
                      auto-rotate
                      camera-controls
                      shadow-intensity="1"
                      environment-image="neutral"
                      style={{ width: '100%', height: '100%' }}
                    />
                  </div>
                  <button className="btn btn-ghost btn-sm" onClick={clearGlb} disabled={running}>
                    Replace
                  </button>
                </div>
              )}
            </div>
          </div>

          <button className="btn btn-primary btn-run" onClick={runPipeline} disabled={!bothReady || running}>
            {running ? (
              <>
                <div className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
                {PHASES[phase] || 'Working'}…
              </>
            ) : bothReady ? (
              'Run validation'
            ) : (
              'Add both files to continue'
            )}
          </button>

          {running && (
            <div className="phase-track">
              {PHASES.map((p, i) => (
                <div
                  key={p}
                  className={`phase-step ${i < phase ? 'done' : ''} ${i === phase ? 'active' : ''}`}
                >
                  <span className="phase-dot" />
                  <span className="phase-name">{p}</span>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* ───────── Results ───────── */}
        {validationResult && (
          <section className="step-section">
            <div className="step-header">
              <div className="step-number">2</div>
              <div>
                <div className="step-title">Validation results</div>
                <div className="step-subtitle">Detailed scoring and analysis</div>
              </div>
            </div>

            {validationResult.evidence && (
              <div className="evidence-panel">
                <div className="evidence-panel__header">
                  <span className="reason-card__title">Validation evidence</span>
                  {validationResult.alignment && (
                    <span className="alignment-badge">
                      Pose az {validationResult.alignment.azimuth}° / el {validationResult.alignment.elevation}°
                      {' · '}IoU {validationResult.alignment.iou}
                      {' · '}conf {Math.round((validationResult.alignment.confidence || 0) * 100)}%
                      {validationResult.alignment.fallback ? ' · fallback' : ''}
                    </span>
                  )}
                </div>
                <div className="evidence-grid">
                  <div className="image-preview">
                    <div className="image-preview__label">Source (background removed)</div>
                    <img src={asset(validationResult.evidence.source_url || bgRemovedUrl)} alt="Source" />
                  </div>
                  <div className="image-preview">
                    <div className="image-preview__label">Aligned render</div>
                    <img src={asset(validationResult.evidence.aligned_render_url || renderedImageUrl)} alt="Aligned render" />
                  </div>
                  <div className="image-preview">
                    <div className="image-preview__label">Overlay</div>
                    {validationResult.evidence.overlay_url ? (
                      <img src={asset(validationResult.evidence.overlay_url)} alt="Overlay comparison" />
                    ) : (
                      <div className="evidence-empty">Overlay unavailable</div>
                    )}
                  </div>
                </div>
              </div>
            )}

            <div className="scores-grid">
              <ScoreRing score={validationResult.geometry.score} label="Geometry" />
              <ScoreRing score={validationResult.texture.score} label="Texture" />
              <ScoreRing score={validationResult.color.score} label="Color" />
            </div>

            <div className="reasons-section">
              <div className="reason-card">
                <div className="reason-card__header">
                  <span className="reason-card__title">Geometry analysis</span>
                </div>
                <p className="reason-card__text">{validationResult.geometry.reason}</p>
                <QualityChips quality={validationResult.geometry.details?.quality_checks} />
                <GateChips gating={validationResult.geometry.details?.gating} />
              </div>

              <div className="reason-card">
                <div className="reason-card__header">
                  <span className="reason-card__title">Texture analysis</span>
                </div>
                <p className="reason-card__text">{validationResult.texture.reason}</p>
                <PresenceChips presence={validationResult.texture.details?.presence_checks} />
                <GateChips gating={validationResult.texture.details?.gating} />
              </div>

              <div className="reason-card">
                <div className="reason-card__header">
                  <span className="reason-card__title">Color analysis</span>
                </div>
                <p className="reason-card__text">{validationResult.color.reason}</p>
                <DominantPalettes dominant={validationResult.color.details?.dominant_color} />
              </div>
            </div>

            {validationResult.performance && <PerformancePanel timings={validationResult.performance} />}
          </section>
        )}

        {status.message && (
          <div className={`status-message ${status.type}`}>
            {status.type === 'info' && <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />}
            {status.message}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
