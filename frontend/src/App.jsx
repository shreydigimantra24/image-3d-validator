import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = '/api';

function ScoreRing({ score, color, label, icon }) {
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;

  const scoreClass = score >= 75 ? 'score-high' : score >= 50 ? 'score-mid' : 'score-low';
  const strokeColor =
    score >= 75 ? 'var(--accent-success)' : score >= 50 ? 'var(--accent-warning)' : 'var(--accent-danger)';

  return (
    <div className="score-card">
      <div className="score-card__icon">{icon}</div>
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

function App() {
  // State
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [imagePath, setImagePath] = useState('');

  const [bgRemovedUrl, setBgRemovedUrl] = useState(null);
  const [bgRemovedPath, setBgRemovedPath] = useState('');

  const [glbFile, setGlbFile] = useState(null);
  const [glbUrl, setGlbUrl] = useState(null);
  const [glbPath, setGlbPath] = useState('');

  const [renderedImageUrl, setRenderedImageUrl] = useState(null);

  const [validationResult, setValidationResult] = useState(null);

  const [loading, setLoading] = useState({ upload: false, bgRemove: false, glbUpload: false, validate: false });
  const [status, setStatus] = useState({ type: '', message: '' });

  const imageInputRef = useRef(null);
  const glbInputRef = useRef(null);

  // ─── Image Upload ───
  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    // Reset downstream state
    setBgRemovedUrl(null);
    setBgRemovedPath('');
    setValidationResult(null);
    setRenderedImageUrl(null);
    setStatus({ type: '', message: '' });
  };

  const uploadImage = async () => {
    if (!imageFile) return;
    setLoading((p) => ({ ...p, upload: true, bgRemove: true }));
    setStatus({ type: 'info', message: 'Uploading image and removing background...' });

    try {
      // Step 1: Upload image
      const formData = new FormData();
      formData.append('file', imageFile);
      const uploadRes = await axios.post(`${API_BASE}/upload/image`, formData);
      const uploadedPath = uploadRes.data.data.filepath;
      setImagePath(uploadedPath);

      // Step 2: Remove background
      const preprocessRes = await axios.post(`${API_BASE}/preprocess`, {
        image_path: uploadedPath,
      });
      const bgResult = preprocessRes.data.data;
      setBgRemovedPath(bgResult.output_path);
      setBgRemovedUrl(bgResult.output_url);
      setStatus({ type: 'success', message: 'Background removed successfully!' });
    } catch (err) {
      console.error(err);
      setStatus({ type: 'error', message: `Failed: ${err.response?.data?.detail || err.message}` });
    } finally {
      setLoading((p) => ({ ...p, upload: false, bgRemove: false }));
    }
  };

  // ─── GLB Upload ───
  const handleGlbSelect = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setGlbFile(file);
    setGlbUrl(URL.createObjectURL(file));
    setValidationResult(null);
    setRenderedImageUrl(null);
  };

  const uploadGlb = async () => {
    if (!glbFile) return;
    setLoading((p) => ({ ...p, glbUpload: true }));
    setStatus({ type: 'info', message: 'Uploading 3D model...' });

    try {
      const formData = new FormData();
      formData.append('file', glbFile);
      const res = await axios.post(`${API_BASE}/upload/glb`, formData);
      setGlbPath(res.data.data.filepath);
      setStatus({ type: 'success', message: '3D model uploaded successfully!' });
    } catch (err) {
      console.error(err);
      setStatus({ type: 'error', message: `GLB upload failed: ${err.response?.data?.detail || err.message}` });
    } finally {
      setLoading((p) => ({ ...p, glbUpload: false }));
    }
  };

  // ─── Validate ───
  const runValidation = async () => {
    if (!bgRemovedPath || !glbPath) return;
    setLoading((p) => ({ ...p, validate: true }));
    setStatus({ type: 'info', message: 'Running validation — this may take a moment...' });
    setValidationResult(null);

    try {
      const res = await axios.post(`${API_BASE}/validate`, {
        image_path: bgRemovedPath,
        glb_path: glbPath,
        original_image_path: imagePath,
      });

      const data = res.data.data;
      setValidationResult(data);
      setRenderedImageUrl(data.rendered_image_url);
      setStatus({ type: 'success', message: 'Validation complete!' });
    } catch (err) {
      console.error(err);
      setStatus({ type: 'error', message: `Validation failed: ${err.response?.data?.detail || err.message}` });
    } finally {
      setLoading((p) => ({ ...p, validate: false }));
    }
  };

  // ─── Drag & Drop helpers ───
  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('active');
  };
  const handleDragLeave = (e) => {
    e.currentTarget.classList.remove('active');
  };
  const handleImageDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('active');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
      setBgRemovedUrl(null);
      setValidationResult(null);
    }
  };
  const handleGlbDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('active');
    const file = e.dataTransfer.files[0];
    if (file && (file.name.endsWith('.glb') || file.name.endsWith('.gltf'))) {
      setGlbFile(file);
      setGlbUrl(URL.createObjectURL(file));
      setValidationResult(null);
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header__badge">
          <span className="header__badge-dot" />
          AI-Powered Pipeline
        </div>
        <h1>Image → 3D Validator</h1>
        <p>Validate 3D models against source product images with precision scoring</p>
      </header>

      <div className="steps">
        {/* ───────── Step 1: Upload Image ───────── */}
        <section className="step-section" id="step-upload-image">
          <div className="step-header">
            <div className="step-number">1</div>
            <div>
              <div className="step-title">Upload Product Image</div>
              <div className="step-subtitle">JPEG, PNG, or WebP — the source for validation</div>
            </div>
          </div>

          {!imagePreview ? (
            <div
              className="upload-zone"
              onClick={() => imageInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleImageDrop}
            >
              <input ref={imageInputRef} type="file" accept="image/*" onChange={handleImageSelect} />
              <div className="upload-zone__icon">📸</div>
              <div className="upload-zone__text">
                <strong>Click to upload</strong> or drag & drop
              </div>
              <div className="upload-zone__hint">JPG, PNG, WebP up to 50MB</div>
            </div>
          ) : (
            <>
              <div className="image-grid">
                <div className="image-preview">
                  <div className="image-preview__label">Original Image</div>
                  <img src={imagePreview} alt="Original product" />
                </div>
                <div className="image-preview">
                  <div className="image-preview__label">Background Removed</div>
                  {bgRemovedUrl ? (
                    <img src={bgRemovedUrl} alt="Background removed" />
                  ) : (
                    <div style={{ height: 280, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                      {loading.bgRemove ? (
                        <div className="loading-overlay">
                          <div className="spinner" />
                          Removing background...
                        </div>
                      ) : (
                        'Click "Process Image" to remove background'
                      )}
                    </div>
                  )}
                </div>
              </div>

              <div className="btn-group">
                <button className="btn btn-primary" onClick={uploadImage} disabled={loading.upload || loading.bgRemove}>
                  {loading.upload || loading.bgRemove ? (
                    <>
                      <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                      Processing...
                    </>
                  ) : (
                    '🚀 Process Image'
                  )}
                </button>
                <button
                  className="btn btn-secondary"
                  onClick={() => {
                    setImageFile(null);
                    setImagePreview(null);
                    setBgRemovedUrl(null);
                    setImagePath('');
                    setBgRemovedPath('');
                    setValidationResult(null);
                  }}
                >
                  ✕ Clear
                </button>
              </div>
            </>
          )}
        </section>

        {/* ───────── Step 2: Upload GLB ───────── */}
        {bgRemovedUrl && (
          <section className="step-section" id="step-upload-glb">
            <div className="step-header">
              <div className="step-number">2</div>
              <div>
                <div className="step-title">Upload 3D Model</div>
                <div className="step-subtitle">GLB or glTF format</div>
              </div>
            </div>

            {!glbUrl ? (
              <div
                className="upload-zone"
                onClick={() => glbInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleGlbDrop}
              >
                <input ref={glbInputRef} type="file" accept=".glb,.gltf" onChange={handleGlbSelect} />
                <div className="upload-zone__icon">🧊</div>
                <div className="upload-zone__text">
                  <strong>Click to upload</strong> or drag & drop a GLB file
                </div>
                <div className="upload-zone__hint">.glb or .gltf format</div>
              </div>
            ) : (
              <>
                <div className="viewer-container">
                  <model-viewer
                    src={glbUrl}
                    alt="3D Model Preview"
                    auto-rotate
                    camera-controls
                    shadow-intensity="1"
                    environment-image="neutral"
                    style={{ width: '100%', height: '100%' }}
                  />
                </div>

                <div className="btn-group">
                  <button className="btn btn-primary" onClick={uploadGlb} disabled={loading.glbUpload || !!glbPath}>
                    {loading.glbUpload ? (
                      <>
                        <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                        Uploading...
                      </>
                    ) : glbPath ? (
                      '✓ Model Uploaded'
                    ) : (
                      '📤 Upload Model'
                    )}
                  </button>
                  <button
                    className="btn btn-secondary"
                    onClick={() => {
                      setGlbFile(null);
                      setGlbUrl(null);
                      setGlbPath('');
                      setValidationResult(null);
                    }}
                  >
                    ✕ Clear Model
                  </button>
                </div>
              </>
            )}
          </section>
        )}

        {/* ───────── Step 3: Validate ───────── */}
        {bgRemovedUrl && glbPath && (
          <section className="step-section" id="step-validate">
            <div className="step-header">
              <div className="step-number">3</div>
              <div>
                <div className="step-title">Run Validation</div>
                <div className="step-subtitle">Geometry, Texture & Color analysis</div>
              </div>
            </div>

            <button
              className="btn btn-primary btn-validate"
              onClick={runValidation}
              disabled={loading.validate}
            >
              {loading.validate ? (
                <>
                  <div className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
                  Analyzing model — please wait...
                </>
              ) : (
                '⚡ Validate 3D Model'
              )}
            </button>
          </section>
        )}

        {/* ───────── Step 4: Results ───────── */}
        {validationResult && (
          <section className="step-section" id="step-results">
            <div className="step-header">
              <div className="step-number">4</div>
              <div>
                <div className="step-title">Validation Results</div>
                <div className="step-subtitle">Detailed scoring and analysis</div>
              </div>
            </div>

            {/* Rendered GLB Preview */}
            {renderedImageUrl && (
              <div className="rendered-preview">
                <div className="image-grid">
                  <div className="image-preview">
                    <div className="image-preview__label">Source (BG Removed)</div>
                    <img src={bgRemovedUrl} alt="Source" />
                  </div>
                  <div className="image-preview">
                    <div className="image-preview__label">Rendered from GLB</div>
                    <img src={renderedImageUrl} alt="Rendered GLB" />
                  </div>
                </div>
              </div>
            )}

            {/* Score Cards */}
            <div className="scores-grid">
              <ScoreRing
                score={validationResult.geometry.score}
                label="Geometry"
                icon="📐"
              />
              <ScoreRing
                score={validationResult.texture.score}
                label="Texture"
                icon="🎨"
              />
              <ScoreRing
                score={validationResult.color.score}
                label="Color"
                icon="🌈"
              />
            </div>

            {/* Detailed Reasons */}
            <div className="reasons-section">
              <div className="reason-card">
                <div className="reason-card__header">
                  <span className="reason-card__icon">📐</span>
                  <span className="reason-card__title">Geometry Analysis</span>
                </div>
                <p className="reason-card__text">{validationResult.geometry.reason}</p>
              </div>

              <div className="reason-card">
                <div className="reason-card__header">
                  <span className="reason-card__icon">🎨</span>
                  <span className="reason-card__title">Texture Analysis</span>
                </div>
                <p className="reason-card__text">{validationResult.texture.reason}</p>
              </div>

              <div className="reason-card">
                <div className="reason-card__header">
                  <span className="reason-card__icon">🌈</span>
                  <span className="reason-card__title">Color Analysis</span>
                </div>
                <p className="reason-card__text">{validationResult.color.reason}</p>
              </div>
            </div>
          </section>
        )}

        {/* Status Bar */}
        {status.message && (
          <div className={`status-message ${status.type}`}>
            {status.type === 'info' && <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />}
            {status.type === 'success' && '✅'}
            {status.type === 'error' && '❌'}
            {status.message}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
