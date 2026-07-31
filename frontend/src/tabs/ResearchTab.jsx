const SWEEP_DATA = [
  { category: "aeroplane", dinov2: 0.098, dino1: 0.049, dinov3: 0.195, agreement: 0.512 },
  { category: "bicycle", dinov2: 0.027, dino1: 0.162, dinov3: 0.459, agreement: 0.162 },
  { category: "bird", dinov2: 0.139, dino1: 0.528, dinov3: 0.417, agreement: 0.639 },
  { category: "boat", dinov2: 0.042, dino1: 0.167, dinov3: 0.292, agreement: 0.458 },
  { category: "bottle", dinov2: 0.029, dino1: 0.235, dinov3: 0.206, agreement: 0.382 },
  { category: "bus", dinov2: 0.026, dino1: 0.128, dinov3: 0.205, agreement: 0.462 },
  { category: "car", dinov2: 0.100, dino1: 0.075, dinov3: 0.225, agreement: 0.300 },
  { category: "cat", dinov2: 0.478, dino1: 0.552, dinov3: 0.388, agreement: 0.806 },
  { category: "chair", dinov2: 0.057, dino1: 0.000, dinov3: 0.200, agreement: 0.286 },
  { category: "cow", dinov2: 0.000, dino1: 0.280, dinov3: 0.540, agreement: 0.440 },
  { category: "dog", dinov2: 0.327, dino1: 0.273, dinov3: 0.564, agreement: 0.600 },
  { category: "horse", dinov2: 0.353, dino1: 0.333, dinov3: 0.451, agreement: 0.647 },
  { category: "motorbike", dinov2: 0.100, dino1: 0.300, dinov3: 0.367, agreement: 0.433 },
  { category: "person", dinov2: -0.073, dino1: 0.415, dinov3: 0.512, agreement: 0.220 },
  { category: "pottedplant", dinov2: 0.037, dino1: 0.074, dinov3: 0.333, agreement: 0.000 },
  { category: "sheep", dinov2: 0.405, dino1: 0.324, dinov3: 0.189, agreement: 0.405 },
  { category: "train", dinov2: 0.082, dino1: 0.066, dinov3: 0.410, agreement: 0.344 },
  { category: "tvmonitor", dinov2: 0.140, dino1: 0.246, dinov3: 0.246, agreement: 0.333 },
];

const SIGNAL_META = {
  dinov2: { label: "DINOv2", color: "#7c9eff" },
  dino1: { label: "DINOv1", color: "#4ade80" },
  dinov3: { label: "DINOv3", color: "#ff9f6b" },
  agreement: { label: "Cross-model agreement", color: "#ff6b4a" },
};

function average(key) {
  return SWEEP_DATA.reduce((sum, row) => sum + row[key], 0) / SWEEP_DATA.length;
}
function negativeCount(key) {
  return SWEEP_DATA.filter((row) => row[key] < 0).length;
}
function winCount(key) {
  return SWEEP_DATA.filter((row) => {
    const best = Math.max(row.dinov2, row.dino1, row.dinov3, row.agreement);
    return row[key] === best;
  }).length;
}

function Bar({ value, max, color }) {
  const pct = Math.max(0, (value / max) * 100);
  const isNegative = value < 0;
  return (
    <div className="bar-track">
      <div
        className={`bar-fill ${isNegative ? "bar-negative" : ""}`}
        style={{ width: `${Math.max(2, Math.abs(pct))}%`, background: isNegative ? "var(--danger)" : color }}
      />
    </div>
  );
}

export default function ResearchTab() {
  const maxGap = Math.max(...SWEEP_DATA.map((r) => Math.max(r.dinov2, r.dino1, r.dinov3, r.agreement)));

  return (
    <>
      <section className="panel">
        <h2>Cross-model agreement as a free uncertainty signal</h2>
        <p className="panel-desc">
          Deep ensembles get uncertainty estimates from disagreement between trained models — the
          expensive part is always training N models. DINOv2, DINOv1, and DINOv3 were never
          trained together, never trained for correspondence, and were built years apart by the
          same lab. Their pairwise disagreement on a predicted point turns out to be a strong,
          <em> zero-training</em> uncertainty signal — often stronger than any single model's own
          self-reported confidence.
        </p>
        <p className="panel-desc">
          Evaluated on all 18 SPair-71k categories, comparing each signal's ability to separate
          correct from incorrect keypoint matches (bucketed top/bottom third by signal value, gap
          = accuracy difference between the buckets — bigger is better).
        </p>
      </section>

      <section className="panel">
        <h2>Headline numbers</h2>
        <div className="stat-row">
          <StatBlock label="avg gap, best individual model" value={Math.max(average("dinov2"), average("dino1"), average("dinov3")).toFixed(3)} />
          <StatBlock label="avg gap, cross-model agreement" value={average("agreement").toFixed(3)} accent="var(--accent-coral)" />
          <StatBlock label="win rate, cross-model agreement" value={`${winCount("agreement")}/18 categories`} accent="var(--accent-coral)" />
        </div>
        <div className="stat-row">
          <StatBlock label="DINOv2 negative categories" value={`${negativeCount("dinov2")}/18`} accent={negativeCount("dinov2") > 0 ? "var(--danger)" : "var(--accent-green)"} />
          <StatBlock label="DINOv1 negative categories" value={`${negativeCount("dino1")}/18`} accent={negativeCount("dino1") > 0 ? "var(--danger)" : "var(--accent-green)"} />
          <StatBlock label="cross-model negative categories" value={`${negativeCount("agreement")}/18`} accent="var(--accent-green)" />
        </div>
      </section>

      <section className="panel">
        <h2>Per-category breakdown</h2>
        <p className="panel-desc">Gap in predicting correctness, by signal. Negative bars (red) mean that signal is actively counterproductive on that category.</p>

        <div className="chart-legend">
          {Object.entries(SIGNAL_META).map(([key, meta]) => (
            <span key={key} className="chart-legend-item">
              <span className="chart-legend-dot" style={{ background: meta.color }} />
              {meta.label}
            </span>
          ))}
        </div>

        <div className="bar-chart">
          {SWEEP_DATA.map((row) => (
            <div key={row.category} className="bar-chart-row">
              <span className="bar-chart-label mono">{row.category}</span>
              <div className="bar-chart-bars">
                {Object.keys(SIGNAL_META).map((key) => (
                  <Bar key={key} value={row[key]} max={maxGap} color={SIGNAL_META[key].color} />
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <h2>Why CLIP dilutes but DINOv3 doesn't</h2>
        <p className="panel-desc">
          Adding CLIP (contrastive image-text training — a fundamentally different objective) as a
          third ensemble member <strong>hurt</strong> the DINOv2+DINOv1 pair: avg gap dropped, win
          rate dropped, and CLIP itself went negative on 2 of 18 categories on its own. Adding
          DINOv3 instead (same self-supervised distillation lineage as v1/v2, despite being a
          different architecture and generation) <strong>helped</strong>: avg gap rose from 0.282 to
          0.413, win rate rose from 9/18 to 12/18, and no model in the trio ever went negative more
          than once.
        </p>
        <p className="panel-desc">
          The takeaway isn't "more models help" — it's that <strong>ensemble members trained under a
          similar paradigm are additive; members trained under a fundamentally different paradigm
          dilute the signal</strong>, even when that different-paradigm model is otherwise strong on
          its own.
        </p>
        <p className="panel-desc" style={{ marginBottom: 0 }}>
          One honest exception: <code className="mono">pottedplant</code> shows cross-model
          agreement at exactly 0.000 — the one category where it provides no separation at all,
          despite individual models doing reasonably. Likely a genuinely ambiguous correspondence
          target (repetitive leaves, few distinctive landmarks).
        </p>
      </section>
    </>
  );
}

function StatBlock({ label, value, accent }) {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <span className="stat-value mono" style={accent ? { color: accent } : undefined}>
        {value}
      </span>
    </div>
  );
}
