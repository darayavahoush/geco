import { useRef, useState } from "react";

const DOT_COLORS = ["#ff6b4a", "#7c9eff", "#4ade80", "#facc15", "#e879f9", "#38bdf8"];

export default function SeedAnnotator({ label, onRemove, file, previewUrl, onFile, keypoints, onKeypointsChange, nameSeed }) {
  const imgRef = useRef(null);
  const inputRef = useRef(null);
  const [pendingName, setPendingName] = useState(nameSeed || "");

  function handleClick(e) {
    if (!previewUrl || !imgRef.current) return;
    const name = pendingName.trim();
    if (!name) return;
    const box = imgRef.current.getBoundingClientRect();
    const x = Math.round(((e.clientX - box.left) / box.width) * 518);
    const y = Math.round(((e.clientY - box.top) / box.height) * 518);
    onKeypointsChange([...keypoints, { name, x, y, displayX: e.clientX - box.left, displayY: e.clientY - box.top }]);
  }

  function removeKeypoint(idx) {
    onKeypointsChange(keypoints.filter((_, i) => i !== idx));
  }

  return (
    <div className="seed-annotator">
      <div className="seed-annotator-head">
        <span className="seed-annotator-label">{label}</span>
        {onRemove && (
          <button className="text-button" onClick={onRemove}>
            remove
          </button>
        )}
      </div>

      {!previewUrl ? (
        <label className="dropzone dropzone-small">
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
          />
          <div className="dropzone-empty" onClick={() => inputRef.current?.click()}>
            <span className="dropzone-icon">⤒</span>
            <span>upload seed image</span>
          </div>
        </label>
      ) : (
        <>
          <div className="seed-annotator-image-wrap" onClick={handleClick}>
            <img ref={imgRef} src={previewUrl} alt={label} />
            {keypoints.map((kp, i) => (
              <div
                key={i}
                className="seed-marker"
                style={{
                  left: kp.displayX,
                  top: kp.displayY,
                  background: DOT_COLORS[i % DOT_COLORS.length],
                }}
                title={kp.name}
              />
            ))}
          </div>
          <div className="seed-annotator-controls">
            <input
              className="text-input"
              placeholder="keypoint name (e.g. left_eye)"
              value={pendingName}
              onChange={(e) => setPendingName(e.target.value)}
            />
          </div>
          {keypoints.length > 0 && (
            <ul className="keypoint-chip-list">
              {keypoints.map((kp, i) => (
                <li key={i} className="keypoint-chip">
                  <span
                    className="keypoint-chip-dot"
                    style={{ background: DOT_COLORS[i % DOT_COLORS.length] }}
                  />
                  <span className="mono">{kp.name}</span>
                  <button className="chip-remove" onClick={() => removeKeypoint(i)}>
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
