export default function ParamSlider({ label, description, value, min, max, step, onChange }) {
  return (
    <label className="param-slider">
      <div className="param-slider-head">
        <span className="param-slider-label">{label}</span>
        <span className="param-slider-value mono">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      {description && <p className="param-slider-desc">{description}</p>}
    </label>
  );
}
