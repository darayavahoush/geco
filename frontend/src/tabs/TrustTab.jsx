import { useRef, useState } from "react";
import Dropzone from "../components/Dropzone.jsx";
import StatCard from "../components/StatCard.jsx";
import { runCrossModelMatch } from "../api";

const MODEL_COLORS = { dinov2: "#7c9eff", dino1: "#4ade80", dinov3: "#ff6b4a", clip: "#facc15" };
const MODEL_LABELS = { dinov2: "DINOv2", dino1: "DINOv1", dinov3: "DINOv3", clip: "CLIP" };

const TRUST_COLOR = { high: "var(--accent-green)", medium: "#facc15", low: "var(--danger)" };

export default function TrustTab() {
  const [srcFile, setSrcFile] = useState(null);
  const [trgFile, setTrgFile] = useState(null);
  const [srcPreview, setSrcPreview] = useState(null);
  const [trgPreview, setTrgPreview] = useState(null);
  const [selectedModels, setSelectedModels] = useState({ dinov2: true, dino1: true, dinov3: true, clip: false });

  const srcImgRef = useRef(null);
  const trgImgRef = useRef(null);

  const [srcMarker, setSrcMarker] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const modelList = Object.entries(selectedModels)
    .filter(([, on]) => on)
    .map(([name]) => name);

  async function handleSrcClick(e) {
    if (!srcFile || !trgFile || !srcImgRef.current) return;
    if (modelList.length < 2) {
      setError("Select at least 2 models to compare.");
      return;
    }
    const box = srcImgRef.current.getBoundingClientRect();
    const dispX = e.clientX - box.left;
    const dispY = e.clientY - box.top;
    setSrcMarker({ x: dispX, y: dispY });

    // Convert displayed (CSS-scaled) click position to the ORIGINAL image's natural pixel space --
    // the backend wants real image coordinates, not whatever size the browser happens to render at.
    const scaleX = srcImgRef.current.naturalWidth / box.width;
    const scaleY = srcImgRef.current.naturalHeight / box.height;
    const pixelX = dispX * scaleX;
    const pixelY = dispY * scaleY;

    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const data = await runCrossModelMatch({
        srcFile,
        trgFile,
        pixelX,
        pixelY,
        models: modelList.join(","),
      });
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function trgDisplayPos(x, y) {
    if (!trgImgRef.current) return { left: 0, top: 0 };
    const box = trgImgRef.current.getBoundingClientRect();
    const scaleX = box.width / trgImgRef.current.naturalWidth;
    const scaleY = box.height / trgImgRef.current.naturalHeight;
    return { left: x * scaleX, top: y * scaleY };
  }

  return (
    <>
      <section className="panel">
        <h2>Cross-Model Trust</h2>
        <p className="panel-desc">
          Click a point on the source image. Each selected backbone independently predicts where
          it lands on the target — no training, no fine-tuning. Agreement between them is a free
          uncertainty signal: tight clustering means high trust, scatter means the match is
          genuinely ambiguous.
        </p>

        <div className="model-toggle-row">
          {Object.keys(selectedModels).map((name) => (
            <label key={name} className="model-toggle">
              <input
                type="checkbox"
                checked={selectedModels[name]}
                onChange={(e) => setSelectedModels({ ...selectedModels, [name]: e.target.checked })}
              />
              <span className="model-toggle-dot" style={{ background: MODEL_COLORS[name] }} />
              {MODEL_LABELS[name]}
            </label>
          ))}
        </div>
        {selectedModels.clip && (
          <p className="model-toggle-warning">
            Note: the 18-category sweep found CLIP dilutes this ensemble (avg gap 0.413 → lower
            when swapped in for DINOv3, win-rate drops accordingly). DINOv3 is additive
            (0.282 → 0.413 avg gap, 9/18 → 12/18 win-rate vs. the DINOv2+DINOv1 pair) — likely
            because it shares DINOv1/v2's self-supervised distillation objective, while CLIP's
            contrastive image-text training is a fundamentally different paradigm.
          </p>
        )}

        <div className="dropzone-row">
          <Dropzone
            label="Source image"
            file={srcFile}
            previewUrl={srcPreview}
            onFile={(file) => {
              setSrcFile(file);
              setSrcPreview(URL.createObjectURL(file));
              setResult(null);
              setSrcMarker(null);
            }}
          />
          <Dropzone
            label="Target image"
            file={trgFile}
            previewUrl={trgPreview}
            onFile={(file) => {
              setTrgFile(file);
              setTrgPreview(URL.createObjectURL(file));
              setResult(null);
              setSrcMarker(null);
            }}
          />
        </div>
      </section>

      {srcFile && trgFile && (
        <section className="panel">
          <h2>Click to compare</h2>
          <div className="correspondence-row">
            <div className="correspondence-image-wrap" onClick={handleSrcClick}>
              <img ref={srcImgRef} src={srcPreview} alt="source" />
              {srcMarker && (
                <div className="marker" style={{ left: srcMarker.x, top: srcMarker.y, background: "#fff" }} />
              )}
            </div>
            <div className="correspondence-image-wrap">
              <img ref={trgImgRef} src={trgPreview} alt="target" />
              {result?.predictions.map((p) => {
                const pos = trgDisplayPos(p.x, p.y);
                return (
                  <div
                    key={p.model}
                    className="marker"
                    style={{ left: pos.left, top: pos.top, background: MODEL_COLORS[p.model] }}
                    title={`${MODEL_LABELS[p.model]}: ${(p.confidence * 100).toFixed(0)}% confidence${p.is_dustbin ? " (dustbin)" : ""}`}
                  />
                );
              })}
            </div>
          </div>

          <div className="correspondence-status">
            {loading && <span className="mono status-pill">running independent models…</span>}
            {error && <span className="mono status-pill status-error">{error}</span>}
            {!loading && !error && !result && (
              <span className="mono status-pill status-muted">click a point on the left image</span>
            )}
          </div>
        </section>
      )}

      {result && (
        <section className="panel">
          <h2>Agreement</h2>
          <div className="stat-row">
            <StatCard
              label="trust"
              value={result.trust.toUpperCase()}
              accent={TRUST_COLOR[result.trust]}
            />
            <StatCard label="mean pairwise disagreement" value={`${result.agreement_px.toFixed(1)} px`} />
            <StatCard label="models compared" value={result.predictions.length} />
          </div>

          <table className="results-table">
            <thead>
              <tr>
                <th>model</th>
                <th>predicted (x, y)</th>
                <th>self-confidence</th>
                <th>status</th>
              </tr>
            </thead>
            <tbody>
              {result.predictions.map((p) => (
                <tr key={p.model}>
                  <td>
                    <span className="keypoint-chip-dot" style={{ background: MODEL_COLORS[p.model], display: "inline-block", marginRight: 6 }} />
                    {MODEL_LABELS[p.model]}
                  </td>
                  <td className="mono">
                    ({p.x.toFixed(0)}, {p.y.toFixed(0)})
                  </td>
                  <td className="mono">{(p.confidence * 100).toFixed(1)}%</td>
                  <td>
                    <span className={`status-pill ${p.is_dustbin ? "status-error" : ""}`}>
                      {p.is_dustbin ? "no match (dustbin)" : "matched"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </>
  );
}
