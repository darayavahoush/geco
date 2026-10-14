// Pose-normalise MediaPipe landmarks onto the canonical face template.
//
// The photo's landmarks (x right, y down, z toward the camera is NEGATIVE, all in units of image width for z) describe a face
// in whatever pose the photo has. The viewer's template is frontal. Mixing posed x/y with frontal template depth shears the
// mesh as soon as the photo is not perfectly frontal. Here we fit a 3D similarity transform (Horn's closed-form
// quaternion method) from the canonical landmarks to the observed ones and map the observation back into the canonical
// (frontal) frame, so x, y AND z come from the person's own geometry with the head pose removed.

function jacobiEigenSym(A, n = 4) {
  const a = A.map((r) => r.slice()); const v = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
    if (off < 1e-22) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]); const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = v[k][p], vkq = v[k][q]; v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq; }
    }
  }
  let best = 0; for (let i = 1; i < n; i++) if (a[i][i] > a[best][best]) best = i;
  return v.map((row) => row[best]);                       // eigenvector of the largest eigenvalue
}

/** Similarity transform dst ~= s * R * src + t. src,dst: arrays of [x,y,z]. Returns { R (3x3 rows), s, t }. */
export function fitSimilarity3D(src, dst) {
  const n = src.length; const mu = (P) => [0, 1, 2].map((k) => P.reduce((a, p) => a + p[k], 0) / n);
  const ms = mu(src), md = mu(dst); const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; let varS = 0;
  for (let i = 0; i < n; i++) {
    const a = [src[i][0] - ms[0], src[i][1] - ms[1], src[i][2] - ms[2]], b = [dst[i][0] - md[0], dst[i][1] - md[1], dst[i][2] - md[2]];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) S[r][c] += a[r] * b[c];
    varS += a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
  }
  const [Sxx, Sxy, Sxz] = S[0], [Syx, Syy, Syz] = S[1], [Szx, Szy, Szz] = S[2];
  const N = [[Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx], [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
             [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy], [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz]];
  let [qw, qx, qy, qz] = jacobiEigenSym(N); const nq = Math.hypot(qw, qx, qy, qz) || 1; qw /= nq; qx /= nq; qy /= nq; qz /= nq;
  const R = [[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
             [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
             [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]];
  let num = 0; for (let i = 0; i < n; i++) {
    const a = [src[i][0] - ms[0], src[i][1] - ms[1], src[i][2] - ms[2]], b = [dst[i][0] - md[0], dst[i][1] - md[1], dst[i][2] - md[2]];
    for (let r = 0; r < 3; r++) num += b[r] * (R[r][0] * a[0] + R[r][1] * a[1] + R[r][2] * a[2]);
  }
  const s = num / (varS || 1);
  const t = [0, 1, 2].map((r) => md[r] - s * (R[r][0] * ms[0] + R[r][1] * ms[1] + R[r][2] * ms[2]));
  return { R, s, t };
}

/**
 * landmarks: MediaPipe [{x,y,z}] normalised to the image (x by width, y by height, z in units of width);
 * canonical: [[x,y,z,...]] x468 template. Returns { verts: [[x,y,z]]x468 in canonical (frontal) units, pose:{yaw,pitch,roll}, s }.
 */
export function frontalizeLandmarks(landmarks, w, h, canonical) {
  const n = 468; const obs = new Array(n), can = new Array(n);
  for (let i = 0; i < n; i++) { const p = landmarks[i]; obs[i] = [p.x * w, -p.y * h, -(p.z || 0) * w]; can[i] = [canonical[i][0], canonical[i][1], canonical[i][2]]; }
  const { R, s, t } = fitSimilarity3D(can, obs);
  const verts = obs.map((o) => { const d = [o[0] - t[0], o[1] - t[1], o[2] - t[2]];
    return [0, 1, 2].map((k) => (R[0][k] * d[0] + R[1][k] * d[1] + R[2][k] * d[2]) / s); });        // R^T (obs - t) / s
  let ss = 0; for (let i = 0; i < n; i++) ss += (verts[i][0] - can[i][0]) ** 2 + (verts[i][1] - can[i][1]) ** 2 + (verts[i][2] - can[i][2]) ** 2;
  const rms = Math.sqrt(ss / n);                      // how far the person's frontal shape is from the template (canonical units)
  const deg = (r) => (r * 180) / Math.PI;
  const pose = { yaw: deg(Math.asin(Math.max(-1, Math.min(1, R[0][2])))), pitch: deg(Math.atan2(-R[1][2], R[2][2])), roll: deg(Math.atan2(-R[0][1], R[0][0])) };
  return { verts, pose, s, rms };
}
