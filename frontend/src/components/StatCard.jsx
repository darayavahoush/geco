export default function StatCard({ label, value, accent }) {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <span className="stat-value mono" style={accent ? { color: accent } : undefined}>
        {value}
      </span>
    </div>
  );
}
