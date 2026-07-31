import { useEffect, useState } from "react";
import OverviewTab from "./tabs/OverviewTab.jsx";
import MatchTab from "./tabs/MatchTab.jsx";
import PropagateTab from "./tabs/PropagateTab.jsx";
import AnomalyTab from "./tabs/AnomalyTab.jsx";
import TrustTab from "./tabs/TrustTab.jsx";
import ResearchTab from "./tabs/ResearchTab.jsx";
import { checkHealth } from "./api";
import "./App.css";

const TABS = [
  { id: "overview", label: "Overview", component: OverviewTab },
  { id: "match", label: "Component 1 — Match", component: MatchTab },
  { id: "propagate", label: "Component 2A — Propagate", component: PropagateTab },
  { id: "anomaly", label: "Component 2B — Anomaly", component: AnomalyTab },
  { id: "trust", label: "Cross-Model Trust", component: TrustTab },
  { id: "research", label: "Research", component: ResearchTab },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [backendStatus, setBackendStatus] = useState("checking");

  useEffect(() => {
    checkHealth()
      .then(() => setBackendStatus("online"))
      .catch(() => setBackendStatus("offline"));
  }, []);

  const ActiveComponent = TABS.find((t) => t.id === activeTab).component;

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header-left" onClick={() => setActiveTab("overview")} style={{ cursor: "pointer" }}>
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

      <nav className="tab-bar">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`tab-button ${activeTab === tab.id ? "tab-active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main className="app-main">
        {activeTab === "overview" ? <ActiveComponent onNavigate={setActiveTab} /> : <ActiveComponent />}
      </main>

      <footer className="app-footer">
        <span>GECO-Enhanced — correspondence engine, propagation, anomaly detection, and cross-model trust.</span>
      </footer>
    </div>
  );
}
