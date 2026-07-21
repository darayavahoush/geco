import { useMemo, useState } from "react";
import Dropzone from "../components/Dropzone.jsx";
import StatCard from "../components/StatCard.jsx";
import { runAnomaly } from "../api";

function verdict(score) {
  if (score < 0.25) return { label: "likely normal", color: "var(--accent-green)" };
  if (score < 0.5) return { label: "borderline", color: "#facc15" };
  return { label: "likely anomalous", color: "var(--danger)" };
}

export default function AnomalyTab() {
  const [referenceFile, setReferenceFile] = useState(null);
  const [referencePreview, setReferencePreview] = useState(null);
  const [testFile, setTestFile] = useState(null);
  const [testPreview, setTestPreview] = useState(null);

  const [status, setStatus] = useState("idle");
  const [errorMsg, setErrorMsg] = useState(null);
  const [result, setResult] = useState(null);

  const canRun = useMemo(() => referenceFile && testFile && status !== "loading", [referenceFile, testFile, status]);

  async function handleRun() {
    setStatus("loading");
    setErrorMsg(null);
    try {
      const data = await runAnomaly({ testFile, referenceFile });
      setResult(data);
      setStatus("done");
    } catch (err) {
      setErrorMsg(err.message);
      setStatus("error");
    }
  }

  const v = result ? verdict(result.anomaly_score) : null;

  return (
    <>
      <section className="panel">
        <h2>1. Golden reference + test image</h2>
        <p className="panel-desc">
          The reference is your known-good example. The test image is scored against it — regions
          that don't correspond well (or get routed to the OT dustbin) are flagged as anomalous.
        </p>
        <div className="dropzone-row">
          <Dropzone
            label="Reference image (known-good)"
            file={referenceFile}
            previewUrl={referencePreview}
            onFile={(file) => {
              setReferenceFile(file);
              setReferencePreview(URL.createObjectURL(file));
              setResult(null);
            }}
          />
          <Dropzone
            label="Test image"
            file={testFile}
            previewUrl={testPreview}
            onFile={(file) => {
              setTestFile(file);
              setTestPreview(URL.createObjectURL(file));
              setResult(null);
            }}
          />
        </div>
        <button className="run-button" disabled={!canRun} onClick={handleRun}>
          {status === "loading" ? "Aligning + scoring…" : "Detect anomalies"}
        </button>
        {status === "error" && <p className="error-text">{errorMsg}</p>}
      </section>

      {result && (
        <section className="panel">
          <h2>2. Result</h2>
          <div className="stat-row">
            <StatCard
              label="anomaly score"
              value={result.anomaly_score.toFixed(3)}
              accent={v.color}
            />
            <StatCard label="verdict" value={v.label} accent={v.color} />
            <StatCard
              label="dustbin patches"
              value={`${result.n_dustbin_patches} / ${result.grid_size * result.grid_size}`}
            />
          </div>

          <div className="viz-grid viz-grid-3">
            <figure>
              <img src={referencePreview} alt="reference" />
              <figcaption>Reference (golden)</figcaption>
            </figure>
            <figure>
              <img src={testPreview} alt="test" />
              <figcaption>Test image</figcaption>
            </figure>
            <figure>
              <img src={`data:image/png;base64,${result.anomaly_heatmap_png}`} alt="anomaly heatmap" />
              <figcaption>Anomaly heatmap (brighter = more anomalous)</figcaption>
            </figure>
          </div>
        </section>
      )}
    </>
  );
}
