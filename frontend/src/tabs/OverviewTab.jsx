const CARDS = [
  {
    id: "match",
    title: "Component 1 — Match",
    desc: "Confidence-aware semantic correspondence between two images, powered by DINOv2 + Optimal Transport.",
  },
  {
    id: "propagate",
    title: "Component 2A — Propagate",
    desc: "Auto-label a large image pool from a handful of hand-annotated seeds. Confidence + cycle-consistency + RANSAC geometric verification.",
  },
  {
    id: "anomaly",
    title: "Component 2B — Anomaly",
    desc: "Reference-based defect detection: hybrid deep + classical scoring against one or more golden references.",
  },
  {
    id: "trust",
    title: "Cross-Model Trust",
    desc: "Zero-training uncertainty from disagreement between independently-trained backbones — DINOv2, DINOv1, DINOv3.",
  },
  {
    id: "face",
    title: "Face Verification",
    desc: "Identity verification powered by frozen DINOv2 ViT-B/14 + ArcFace margin projection head (99.40% benchmark accuracy).",
  },
];

export default function OverviewTab({ onNavigate }) {
  return (
    <>
      <section className="hero">
        <span className="hero-eyebrow mono">DINOv2 · Optimal Transport · Multi-Model Ensembles</span>
        <h1 className="hero-title">Confidence-aware correspondence, and knowing when to trust it</h1>
        <p className="hero-desc">
          A semantic correspondence engine built on DINOv2 and confidence-weighted Optimal
          Transport, extended into two applications — automated annotation propagation and
          reference-based anomaly detection — plus a research finding: agreement between
          independently-trained foundation models is a free, training-free uncertainty signal
          that beats any single model's own introspection.
        </p>
      </section>

      <section className="panel">
        <h2>Headline result</h2>
        <p className="panel-desc">
          Across all 18 SPair-71k categories, cross-model agreement (DINOv2+DINOv1+DINOv3) predicts
          correspondence correctness better than any individual model's self-confidence — and
          never once goes actively counterproductive.
        </p>
        <div className="stat-row">
          <div className="stat-card">
            <span className="stat-label">win rate vs. every individual model</span>
            <span className="stat-value mono" style={{ color: "var(--accent-coral)" }}>12 / 18 categories</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">categories where agreement fails</span>
            <span className="stat-value mono" style={{ color: "var(--accent-green)" }}>0 / 18</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">avg. gap vs. best single model</span>
            <span className="stat-value mono">0.413 vs 0.282</span>
          </div>
        </div>
        <button className="secondary-button" onClick={() => onNavigate("research")}>
          See the full research breakdown →
        </button>
      </section>

      <section className="panel">
        <h2>Explore the tools</h2>
        <div className="card-grid">
          {CARDS.map((card) => (
            <button key={card.id} className="nav-card" onClick={() => onNavigate(card.id)}>
              <h3>{card.title}</h3>
              <p>{card.desc}</p>
              <span className="nav-card-cta">Open →</span>
            </button>
          ))}
        </div>
      </section>
    </>
  );
}
