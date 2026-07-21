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

// ── Component 2 / Track A — annotation propagation ──────────────────────────
export async function runPropagate({ targetFile, seeds, confidenceThreshold, cycleErrorThreshold }) {
  const form = new FormData();
  form.append("target_image", targetFile);
  for (const seed of seeds) {
    form.append("seed_images", seed.file);
    form.append(
      "seed_keypoints_json",
      JSON.stringify(seed.keypoints.map((k) => ({ name: k.name, x: k.x, y: k.y })))
    );
  }
  form.append("confidence_threshold", confidenceThreshold);
  form.append("cycle_error_threshold", cycleErrorThreshold);

  const res = await fetch(`${API_URL}/api/propagate`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Propagation failed (${res.status}): ${detail}`);
  }
  return res.json();
}

// ── Component 2 / Track B — anomaly detection ────────────────────────────────
export async function runAnomaly({ testFile, referenceFile }) {
  const form = new FormData();
  form.append("test_image", testFile);
  form.append("reference_image", referenceFile);

  const res = await fetch(`${API_URL}/api/anomaly`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Anomaly detection failed (${res.status}): ${detail}`);
  }
  return res.json();
}
