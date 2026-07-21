import { useMemo, useState } from "react";
import Dropzone from "../components/Dropzone.jsx";
import ParamSlider from "../components/ParamSlider.jsx";
import StatCard from "../components/StatCard.jsx";
import SeedAnnotator from "../components/SeedAnnotator.jsx";
import { runPropagate } from "../api";

let seedIdCounter = 0;

export default function PropagateTab() {
  const [seeds, setSeeds] = useState([{ id: seedIdCounter++, file: null, preview: null, keypoints: [] }]);
  const [targetFile, setTargetFile] = useState(null);
  const [targetPreview, setTargetPreview] = useState(null);

  const [confidenceThreshold, setConfidenceThreshold] = useState(0.15);
  const [cycleErrorThreshold, setCycleErrorThreshold] = useState(40);

  const [status, setStatus] = useState("idle");
  const [errorMsg, setErrorMsg] = useState(null);
  const [results, setResults] = useState(null);

  function addSeed() {
    setSeeds([...seeds, { id: seedIdCounter++, file: null, preview: null, keypoints: [] }]);
  }
  function removeSeed(id) {
    setSeeds(seeds.filter((s) => s.id !== id));
  }
  function updateSeed(id, patch) {
    setSeeds(seeds.map((s) => (s.id === id ? { ...s, ...patch } : s)));
  }

  const totalKeypointNames = useMemo(() => {
    const names = new Set();
    seeds.forEach((s) => s.keypoints.forEach((k) => names.add(k.name)));
    return names.size;
  }, [seeds]);

  const canRun = useMemo(
    () =>
      targetFile &&
      seeds.some((s) => s.file && s.keypoints.length > 0) &&
      status !== "loading",
    [targetFile, seeds, status]
  );

  async function handleRun() {
    setStatus("loading");
    setErrorMsg(null);
    try {
      const validSeeds = seeds.filter((s) => s.file && s.keypoints.length > 0);
      const data = await runPropagate({
        targetFile,
        seeds: validSeeds,
        confidenceThreshold,
        cycleErrorThreshold,
      });
      setResults(data);
      setStatus("done");
    } catch (err) {
      setErrorMsg(err.message);
      setStatus("error");
    }
  }

  const acceptedCount = results?.filter((r) => r.accepted).length ?? 0;

  return (
    <>
      <section className="panel">
        <h2>1. Annotate seed images</h2>
        <p className="panel-desc">
          Click a point on each seed image after typing a keypoint name (e.g. <code className="mono">left_eye</code>).
          Use the same names across seeds so votes can be pooled per keypoint.
        </p>
        <div className="seed-grid">
          {seeds.map((seed) => (
            <SeedAnnotator
              key={seed.id}
              label={`Seed ${seeds.indexOf(seed) + 1}`}
              file={seed.file}
              previewUrl={seed.preview}
              keypoints={seed.keypoints}
              onFile={(file) => updateSeed(seed.id, { file, preview: URL.createObjectURL(file) })}
              onKeypointsChange={(kps) => updateSeed(seed.id, { keypoints: kps })}
              onRemove={seeds.length > 1 ? () => removeSeed(seed.id) : undefined}
            />
          ))}
        </div>
        <button className="secondary-button" onClick={addSeed}>
          + add another seed image
        </button>
      </section>

      <section className="panel">
        <h2>2. Target image (unlabeled)</h2>
        <p className="panel-desc">Keypoints from every seed above will be propagated onto this image.</p>
        <Dropzone
          label="Target image"
          file={targetFile}
          previewUrl={targetPreview}
          onFile={(file) => {
            setTargetFile(file);
            setTargetPreview(URL.createObjectURL(file));
            setResults(null);
          }}
        />
      </section>

      <section className="panel">
        <h2>3. Acceptance thresholds</h2>
        <ParamSlider
          label="confidence threshold"
          description="Minimum OT transport confidence for a propagated point to be accepted."
          value={confidenceThreshold}
          min={0}
          max={0.6}
          step={0.01}
          onChange={setConfidenceThreshold}
        />
        <ParamSlider
          label="cycle-error threshold (px, 518-space)"
          description="Max allowed round-trip distance (target→seed→target) before a point is rejected."
          value={cycleErrorThreshold}
          min={5}
          max={120}
          step={1}
          onChange={setCycleErrorThreshold}
        />
        <button className="run-button" disabled={!canRun} onClick={handleRun}>
          {status === "loading"
            ? `Propagating ${totalKeypointNames} keypoint${totalKeypointNames === 1 ? "" : "s"}…`
            : "Propagate keypoints"}
        </button>
        {status === "error" && <p className="error-text">{errorMsg}</p>}
      </section>

      {results && (
        <section className="panel">
          <h2>4. Propagated result</h2>
          <div className="stat-row">
            <StatCard label="keypoints propagated" value={results.length} />
            <StatCard label="accepted" value={acceptedCount} accent="var(--accent-green)" />
            <StatCard
              label="rejected"
              value={results.length - acceptedCount}
              accent="var(--danger)"
            />
          </div>

          <div className="propagate-result-wrap">
            <img src={targetPreview} alt="target with propagated keypoints" />
            {results.map((r, i) => (
              <div
                key={i}
                className={`prop-marker ${r.accepted ? "prop-accepted" : "prop-rejected"}`}
                style={{ left: `${(r.x / 518) * 100}%`, top: `${(r.y / 518) * 100}%` }}
                title={`${r.name} — conf ${(r.confidence * 100).toFixed(0)}%, cycle err ${r.cycle_error.toFixed(1)}px`}
              />
            ))}
          </div>

          <table className="results-table">
            <thead>
              <tr>
                <th>keypoint</th>
                <th>confidence</th>
                <th>cycle error (px)</th>
                <th>votes</th>
                <th>status</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => (
                <tr key={i}>
                  <td className="mono">{r.name}</td>
                  <td className="mono">{(r.confidence * 100).toFixed(1)}%</td>
                  <td className="mono">{r.cycle_error.toFixed(1)}</td>
                  <td className="mono">{r.n_votes}</td>
                  <td>
                    <span className={`status-pill ${r.accepted ? "" : "status-error"}`}>
                      {r.accepted ? "accepted" : "rejected"}
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
