import { useEffect, useMemo, useState } from "react";
import Dropzone from "./components/Dropzone.jsx";
import ParamSlider from "./components/ParamSlider.jsx";
import StatCard from "./components/StatCard.jsx";
import CorrespondenceCanvas from "./components/CorrespondenceCanvas.jsx";
import { runMatch, checkHealth } from "./api";
import "./App.css";

const IMAGE_SIZE = 518; // native DINOv2 input resolution used by the backend

export default function App() {
  const [srcFile, setSrcFile] = useState(null);
  const [trgFile, setTrgFile] = useState(null);
  const [srcPreview, setSrcPreview] = useState(null);
  const [trgPreview, setTrgPreview] = useState(null);

  const [alpha, setAlpha] = useState(0.8);
  const [zBase, setZBase] = useState(0.3);
  const [zRange, setZRange] = useState(0.25);
  const [reg, setReg] = useState(0.05);

  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [errorMsg, setErrorMsg] = useState(null);
  const [result, setResult] = useState(null);
  const [backendStatus, setBackendStatus] = useState("checking");

  useEffect(() => {
    checkHealth()
      .then(() => setBackendStatus("online"))
      .catch(() => setBackendStatus("offline"));
  }, []);

  function handleSrcFile(file) {
    setSrcFile(file);
    setSrcPreview(URL.createObjectURL(file));
    setResult(null);
  }
  function handleTrgFile(file) {
    setTrgFile(file);
    setTrgPreview(URL.createObjectURL(file));
    setResult(null);
  }

  const canRun = useMemo(() => srcFile && trgFile && status !== "loading", [srcFile, trgFile, status]);

  async function handleRun() {
    setStatus("loading");
    setErrorMsg(null);
    try {
      const data = await runMatch({ srcFile, trgFile, alpha, zBase, zRange, reg });
      setResult(data);
      setStatus("done");
    } catch (err) {
      setErrorMsg(err.message);
      setStatus("error");
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header-left">
          <span className="logo-mark">GC</span>
          <div>
            <h1>GECO</h1>
            <p className="app-subtitle">confidence-aware semantic correspondence · DINOv2 + optimal transport</p>
          </div>
        </div>
        <div className={`backend-pill mono status-${backendStatus}`}>
          <span className="dot" /> backend: {backendStatus}
        </div>
      </header>

      <main className="app-main">
        <section className="panel panel-inputs">
          <h2>1. Choose two images</h2>
          <p className="panel-desc">
            Same object category, different pose or instance — e.g. two different birds, or the
            same bird from two angles.
          </p>
          <div className="dropzone-row">
            <Dropzone label="Source image" file={srcFile} previewUrl={srcPreview} onFile={handleSrcFile} />
            <Dropzone label="Target image" file={trgFile} previewUrl={trgPreview} onFile={handleTrgFile} />
          </div>
        </section>

        <section className="panel panel-params">
          <h2>2. Tune the matching pipeline</h2>
          <ParamSlider
            label="alpha — confidence weighting"
            description="0 = vanilla uniform OT marginals, 1 = fully confidence-weighted matching."
            value={alpha}
            min={0}
            max={1}
            step={0.05}
            onChange={setAlpha}
          />
          <ParamSlider
            label="z_base — dustbin base score"
            description="Baseline cost of declaring 'no match' for a maximally confident patch."
            value={zBase}
            min={0.1}
            max={0.6}
            step={0.01}
            onChange={setZBase}
          />
          <ParamSlider
            label="z_range — adaptive dustbin range"
            description="How much low-confidence patches get an easier path to the dustbin."
            value={zRange}
            min={0}
            max={0.4}
            step={0.01}
            onChange={setZRange}
          />
          <ParamSlider
            label="reg — Sinkhorn regularization"
            description="Lower = sharper assignments but slower/less stable convergence."
            value={reg}
            min={0.01}
            max={0.3}
            step={0.01}
            onChange={setReg}
          />
          <button className="run-button" disabled={!canRun} onClick={handleRun}>
            {status === "loading" ? "Running Sinkhorn matching…" : "Run matching"}
          </button>
          {status === "error" && <p className="error-text">{errorMsg}</p>}
        </section>

        {result && (
          <>
            <section className="panel">
              <h2>3. Click to transfer a keypoint</h2>
              <p className="panel-desc">
                Click any point on the source image — the matched point on the target is found via
                the transport plan computed above.
              </p>
              <CorrespondenceCanvas
                srcPreview={srcPreview}
                trgPreview={trgPreview}
                sessionId={result.session_id}
                imageSize={IMAGE_SIZE}
              />
            </section>

            <section className="panel">
              <h2>4. Pipeline internals</h2>
              <div className="stat-row">
                <StatCard label="patch grid" value={`${result.grid_size} × ${result.grid_size}`} />
                <StatCard
                  label="transport entropy"
                  value={result.entropy.toFixed(3)}
                  accent="var(--accent-blue)"
                />
                <StatCard
                  label="mean match confidence"
                  value={`${(result.mean_confidence * 100).toFixed(1)}%`}
                  accent="var(--accent-coral)"
                />
              </div>

              <div className="viz-grid">
                <figure>
                  <img src={`data:image/png;base64,${result.src_confidence_png}`} alt="source confidence" />
                  <figcaption>Source patch confidence</figcaption>
                </figure>
                <figure>
                  <img src={`data:image/png;base64,${result.trg_confidence_png}`} alt="target confidence" />
                  <figcaption>Target patch confidence</figcaption>
                </figure>
                <figure>
                  <img src={`data:image/png;base64,${result.src_pca_png}`} alt="source PCA features" />
                  <figcaption>Source features (PCA → RGB)</figcaption>
                </figure>
                <figure>
                  <img src={`data:image/png;base64,${result.trg_pca_png}`} alt="target PCA features" />
                  <figcaption>Target features (PCA → RGB)</figcaption>
                </figure>
              </div>
            </section>
          </>
        )}
      </main>

      <footer className="app-footer">
        <span>GECO-Enhanced — capstone build on DINOv2 multi-scale features + confidence-aware OT.</span>
      </footer>
    </div>
  );
}
