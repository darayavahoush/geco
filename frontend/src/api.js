const API_URL = import.meta.env.VITE_API_URL || "";

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
  form.append("reference_images", referenceFile);

  const res = await fetch(`${API_URL}/api/anomaly`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Anomaly detection failed (${res.status}): ${detail}`);
  }
  return res.json();
}

// ── Face Verification — frozen DINOv2 + trained projection head, identity task ──
export async function runVerifyFace({ photoAFile, photoBFile, threshold = null }) {
  const form = new FormData();
  form.append("photo_a", photoAFile);
  form.append("photo_b", photoBFile);
  if (threshold !== null && threshold !== undefined) {
    form.append("threshold", threshold);
  }

  const res = await fetch(`${API_URL}/api/verify-face`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Face verification failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function fetchFaceSamples() {
  const res = await fetch(`${API_URL}/api/face-samples`);
  if (!res.ok) {
    throw new Error(`Failed to load face samples (${res.status})`);
  }
  return res.json();
}

export async function fetchFaceModelInfo() {
  const res = await fetch(`${API_URL}/api/face-model-info`);
  if (!res.ok) {
    throw new Error(`Failed to load face model info (${res.status})`);
  }
  return res.json();
}

// ── Cross-Model Trust — DINOv2/DINOv1(/CLIP) agreement as a free uncertainty signal ──
export async function runCrossModelMatch({ srcFile, trgFile, pixelX, pixelY, models = "dinov2,dino1" }) {
  const form = new FormData();
  form.append("src_image", srcFile);
  form.append("trg_image", trgFile);
  form.append("pixel_x", pixelX);
  form.append("pixel_y", pixelY);
  form.append("models", models);

  const res = await fetch(`${API_URL}/api/cross-model-match`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Cross-model match failed (${res.status}): ${detail}`);
  }
  return res.json();
}
