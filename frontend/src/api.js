const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function runMatch({ srcFile, trgFile, alpha, zBase, zRange, reg }) {
  const form = new FormData();
  form.append("src_image", srcFile);
  form.append("trg_image", trgFile);
  form.append("alpha", alpha);
  form.append("z_base", zBase);
  form.append("z_range", zRange);
  form.append("reg", reg);

  const res = await fetch(`${API_URL}/api/match`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Matching failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function getKeypointMatch({ sessionId, pixelX, pixelY, imageSize }) {
  const res = await fetch(`${API_URL}/api/keypoint`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      pixel_x: pixelX,
      pixel_y: pixelY,
      image_size: imageSize,
    }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Keypoint lookup failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function checkHealth() {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error("Backend unreachable");
  return res.json();
}
