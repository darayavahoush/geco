import { useLayoutEffect, useRef, useState } from "react";
import { getKeypointMatch } from "../api";

const GRID_CELLS = 12; // decorative subdivision shown to the user (native model grid is 37x37)

function GridOverlay() {
  const lines = [];
  for (let i = 1; i < GRID_CELLS; i++) {
    const pct = (i / GRID_CELLS) * 100;
    lines.push(
      <line key={`v${i}`} x1={`${pct}%`} y1="0" x2={`${pct}%`} y2="100%" />,
      <line key={`h${i}`} x1="0" y1={`${pct}%`} x2="100%" y2={`${pct}%`} />
    );
  }
  return (
    <svg className="grid-overlay" preserveAspectRatio="none">
      {lines}
    </svg>
  );
}

export default function CorrespondenceCanvas({ srcPreview, trgPreview, sessionId, imageSize }) {
  const rowRef = useRef(null);
  const srcImgRef = useRef(null);
  const trgImgRef = useRef(null);
  const [srcMarker, setSrcMarker] = useState(null); // {x,y} in displayed px, relative to its own image
  const [trgMarker, setTrgMarker] = useState(null);
  const [linePath, setLinePath] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastMatch, setLastMatch] = useState(null);

  useLayoutEffect(() => {
    if (!srcMarker || !trgMarker || !rowRef.current || !srcImgRef.current || !trgImgRef.current) {
      setLinePath(null);
      return;
    }
    const rowBox = rowRef.current.getBoundingClientRect();
    const srcBox = srcImgRef.current.getBoundingClientRect();
    const trgBox = trgImgRef.current.getBoundingClientRect();

    const x1 = srcBox.left - rowBox.left + srcMarker.x;
    const y1 = srcBox.top - rowBox.top + srcMarker.y;
    const x2 = trgBox.left - rowBox.left + trgMarker.x;
    const y2 = trgBox.top - rowBox.top + trgMarker.y;
    const midX = (x1 + x2) / 2;

    setLinePath({ x1, y1, x2, y2, midX });
  }, [srcMarker, trgMarker]);

  async function handleSrcClick(e) {
    if (!sessionId || !srcImgRef.current) return;
    const box = srcImgRef.current.getBoundingClientRect();
    const x = e.clientX - box.left;
    const y = e.clientY - box.top;
    setSrcMarker({ x, y });
    setTrgMarker(null);
    setError(null);
    setLoading(true);
    try {
      const pixelX = Math.round((x / box.width) * imageSize);
      const pixelY = Math.round((y / box.height) * imageSize);
      const result = await getKeypointMatch({ sessionId, pixelX, pixelY, imageSize });
      const trgBox = trgImgRef.current.getBoundingClientRect();
      setTrgMarker({
        x: (result.trg_pixel_x / imageSize) * trgBox.width,
        y: (result.trg_pixel_y / imageSize) * trgBox.height,
      });
      setLastMatch(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="correspondence-block">
      <div className="correspondence-row" ref={rowRef}>
        <div className="correspondence-image-wrap">
          <img ref={srcImgRef} src={srcPreview} alt="source" onClick={handleSrcClick} />
          <GridOverlay />
          {srcMarker && (
            <div className="marker marker-src" style={{ left: srcMarker.x, top: srcMarker.y }} />
          )}
        </div>

        <svg className="connector-svg">
          {linePath && (
            <path
              className="connector-path"
              d={`M ${linePath.x1} ${linePath.y1} C ${linePath.midX} ${linePath.y1}, ${linePath.midX} ${linePath.y2}, ${linePath.x2} ${linePath.y2}`}
            />
          )}
        </svg>

        <div className="correspondence-image-wrap">
          <img ref={trgImgRef} src={trgPreview} alt="target" />
          <GridOverlay />
          {trgMarker && (
            <div
              className={`marker marker-trg ${lastMatch?.is_dustbin ? "marker-dustbin" : ""}`}
              style={{ left: trgMarker.x, top: trgMarker.y }}
            />
          )}
        </div>
      </div>

      <div className="correspondence-status">
        {loading && <span className="mono status-pill">locating match…</span>}
        {error && <span className="mono status-pill status-error">{error}</span>}
        {!loading && !error && lastMatch && (
          <span className="mono status-pill">
            {lastMatch.is_dustbin
              ? "no confident match — routed to dustbin"
              : `match confidence: ${(lastMatch.confidence * 100).toFixed(1)}%`}
          </span>
        )}
        {!loading && !error && !lastMatch && (
          <span className="mono status-pill status-muted">click a point on the left image</span>
        )}
      </div>
    </div>
  );
}
