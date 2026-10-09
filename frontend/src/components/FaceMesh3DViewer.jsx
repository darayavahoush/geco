import { useEffect, useRef, useState, useCallback, useMemo } from "react";

/**
 * Builds a continuous, high-density 3D anatomical human head/face mesh.
 * Adapts to detected facial landmarks (pupil distance, nose protrusion, jaw width)
 * and conforms to natural human facial curves (forehead slope, nose bridge/tip,
 * eye socket dips, lips, and chin) without hard-edged mask boundaries or origami folds.
 */
export function buildAnatomicalFaceMesh(faceMeta = null, coveredAngles = ["front"], nTheta = 24, nPhi = 32) {
  const covered = new Set(coveredAngles || ["front"]);

  let scaleEyeW = 1.0;
  let shiftEyeY = 0.0;
  let shiftNoseY = 0.0;
  let scaleNoseZ = 1.0;
  let shiftMouthY = 0.0;
  let scaleMouthW = 1.0;
  let scaleJawW = 1.0;
  let scaleFaceH = 1.0;

  if (faceMeta && faceMeta.landmarks && faceMeta.bbox) {
    const lm = faceMeta.landmarks;
    const [bx, by, bw, bh] = faceMeta.bbox;
    const cx = bx + bw / 2.0;
    const cy = by + bh / 2.0;

    const le = lm.left_eye || [bx + bw * 0.34, by + bh * 0.38];
    const re = lm.right_eye || [bx + bw * 0.66, by + bh * 0.38];
    const nt = lm.nose_tip || [bx + bw * 0.50, by + bh * 0.55];
    const lMouth = lm.left_mouth || [bx + bw * 0.36, by + bh * 0.72];
    const rMouth = lm.right_mouth || [bx + bw * 0.64, by + bh * 0.72];

    const iodPx = Math.abs(re[0] - le[0]);
    scaleEyeW = Math.max(0.85, Math.min(1.20, (iodPx / Math.max(1.0, bw)) / 0.42));

    const eyeYNorm = ((le[1] + re[1]) / 2.0 - cy) / Math.max(1.0, bh);
    shiftEyeY = (-eyeYNorm - 0.20) * 0.25;

    const noseYNorm = (nt[1] - cy) / Math.max(1.0, bh);
    shiftNoseY = (-noseYNorm - -0.10) * 0.30;

    const mouthYNorm = ((lMouth[1] + rMouth[1]) / 2.0 - cy) / Math.max(1.0, bh);
    shiftMouthY = (-mouthYNorm - -0.35) * 0.25;

    const mouthWPx = Math.abs(rMouth[0] - lMouth[0]);
    scaleMouthW = Math.max(0.85, Math.min(1.20, (mouthWPx / Math.max(1.0, bw)) / 0.38));

    const aspect = bw / Math.max(1.0, bh);
    scaleJawW = Math.max(0.85, Math.min(1.15, aspect / 0.85));
    scaleFaceH = Math.max(0.90, Math.min(1.15, 0.85 / Math.max(0.5, aspect)));
  }

  // Grid of latitudes theta (from 0.12 at top crown to 0.92*pi at chin/neck)
  const thetas = [];
  for (let j = 0; j < nTheta; j++) {
    thetas.push(0.12 + (Math.PI * 0.80 * j) / (nTheta - 1));
  }
  // Grid of longitudes phi (-pi to pi)
  const phis = [];
  for (let i = 0; i < nPhi; i++) {
    phis.push(-Math.PI + (2.0 * Math.PI * i) / nPhi);
  }

  const phiFace = 1.25; // ~71.6 degrees: facial frontal span

  const vertices = [];
  const uvs = [];
  const confidences = [];
  const isCranial = [];

  for (let j = 0; j < nTheta; j++) {
    const t = thetas[j];
    const rawY = Math.cos(t) * 0.68;
    const y = Math.round(rawY * scaleFaceH * 10000) / 10000;

    // Head cross-section radii at vertical level y
    let rx, rzFront, rzBack;
    if (y >= 0) {
      const shapeFactor = Math.sqrt(Math.max(0.04, 1.0 - Math.pow(y / 0.72, 2)));
      rx = 0.48 * shapeFactor;
      rzFront = 0.28 * shapeFactor;
      rzBack = 0.40 * shapeFactor;
    } else {
      const prog = -y / 0.65;
      rx = 0.48 * (1.0 - 0.32 * Math.pow(prog, 1.2)) * scaleJawW;
      rzFront = 0.28 * (1.0 - 0.20 * prog);
      rzBack = 0.40 * (1.0 - 0.45 * prog);
    }

    for (let i = 0; i < nPhi; i++) {
      const p = phis[i];
      const sinP = Math.sin(p);
      const cosP = Math.cos(p);

      const x = Math.round(rx * sinP * 10000) / 10000;
      const isFront = cosP >= 0;
      const zBase = (isFront ? rzFront : rzBack) * cosP;

      // Sagittal facial features (only on the front face)
      let zRelief = 0.0;
      const isFace = Math.abs(p) <= phiFace && y >= -0.62 && y <= 0.52;

      if (isFace) {
        // 1. 3D Anatomical Nose (bridge down to tip): realistic anthropometric protrusion (0.075 max)
        const noseYMin = -0.16 + shiftNoseY;
        const noseYMax = 0.14 + shiftNoseY;
        if (y >= noseYMin && y <= noseYMax) {
          const latN = Math.exp(-0.5 * Math.pow(p / 0.12, 2));
          const noseTipY = -0.06 + shiftNoseY;
          let vertN = 0.0;
          if (y >= noseTipY) {
            vertN = 0.03 + 0.045 * ((noseYMax - y) / Math.max(0.01, noseYMax - noseTipY));
          } else {
            vertN = 0.03 + 0.045 * ((y - noseYMin) / Math.max(0.01, noseTipY - noseYMin));
          }
          zRelief += vertN * latN * Math.min(1.20, Math.max(0.80, scaleNoseZ));
        }

        // 2. Orbits (eye sockets centered at y = 0.12)
        const eyeY = 0.12 + shiftEyeY;
        const dEye = Math.sqrt(Math.pow((Math.abs(p) - 0.35 * scaleEyeW) / 0.20, 2) + Math.pow((y - eyeY) / 0.12, 2));
        if (dEye < 1.0) {
          zRelief += -0.020 * (1.0 - dEye * dEye);
        }

        // 3. Brow ridge at y = 0.22
        const browY = 0.22 + shiftEyeY;
        if (Math.abs(y - browY) < 0.08 && Math.abs(p) < 0.48) {
          zRelief += 0.016 * Math.exp(-0.5 * Math.pow(p / 0.35, 2)) * (1.0 - Math.abs(y - browY) / 0.08);
        }

        // 4. Lips centered at y = -0.25
        const mouthYMin = -0.32 + shiftMouthY;
        const mouthYMax = -0.18 + shiftMouthY;
        if (y >= mouthYMin && y <= mouthYMax && Math.abs(p) < 0.30) {
          const latM = Math.exp(-0.5 * Math.pow(p / (0.20 * scaleMouthW), 2));
          zRelief += 0.024 * Math.sin(((y - mouthYMin) / 0.14) * Math.PI) * latM;
        }

        // 5. Chin (pogonion) centered at y = -0.46
        const chinYMin = -0.54;
        const chinYMax = -0.38;
        if (y >= chinYMin && y <= chinYMax && Math.abs(p) < 0.25) {
          const latC = Math.exp(-0.5 * Math.pow(p / 0.16, 2));
          zRelief += 0.030 * Math.sin(((y - chinYMin) / 0.16) * Math.PI) * latC;
        }
      }

      const z = Math.round((zBase + zRelief) * 10000) / 10000;
      vertices.push([x, y, z]);
      isCranial.push(!isFace);

      // UV mapping calibrated to canonical facial proportions
      if (isFace) {
        let u = 0.50 + 0.45 * p;
        let v = 0.55 - 0.75 * y;
        u = Math.max(0.01, Math.min(0.99, u));
        v = Math.max(0.01, Math.min(0.99, v));
        uvs.push({ u: Math.round(u * 10000) / 10000, v: Math.round(v * 10000) / 10000 });
      } else {
        uvs.push({ u: 0.5, v: 0.5 });
      }

      // Confidence
      let conf = 0.95;
      if (!covered.has("front")) conf = 0.40;
      if (Math.abs(p) > phiFace * 0.70) {
        const sideCovered = (covered.has("left") && p < 0) || (covered.has("right") && p > 0);
        conf = sideCovered ? 0.95 : 0.45;
      }
      confidences.push(Math.round(conf * 100) / 100);
    }
  }

  // Regular CCW quad-split triangles
  const triangles = [];
  for (let j = 0; j < nTheta - 1; j++) {
    for (let i = 0; i < nPhi; i++) {
      const iNext = (i + 1) % nPhi;
      const v00 = j * nPhi + i;
      const v10 = j * nPhi + iNext;
      const v01 = (j + 1) * nPhi + i;
      const v11 = (j + 1) * nPhi + iNext;

      triangles.push([v00, v01, v10]);
      triangles.push([v10, v01, v11]);
    }
  }

  return { vertices, triangles, uvs, confidences, isCranial };
}

// Detect face sub-region in source photo (isolating face from shoulders and background)
function getFaceCropRect(img, faceMeta) {
  if (!img || !img.width || !img.height) return null;
  const aspect = img.width / img.height;
  // If the image is already a tight face crop (e.g., from YuNet extraction or square crop), use it directly
  if (aspect >= 0.92 && aspect <= 1.08 && img.width <= 320) {
    return { sx: 0, sy: 0, sw: img.width, sh: img.height };
  }
  if (faceMeta && faceMeta.bbox && img.width > faceMeta.bbox[2] * 1.05) {
    const [bx, by, bw, bh] = faceMeta.bbox;
    const margin = Math.max(bw, bh) * 0.18;
    const sx = Math.max(0, Math.round(bx - margin));
    const sy = Math.max(0, Math.round(by - margin * 1.1));
    const sw = Math.min(img.width - sx, Math.round(bw + margin * 2));
    const sh = Math.min(img.height - sy, Math.round(bh + margin * 2.2));
    const side = Math.min(sw, sh);
    return { sx, sy, sw: side, sh: side };
  }
  if (aspect >= 0.92 && aspect <= 1.08) {
    return { sx: 0, sy: 0, sw: img.width, sh: img.height };
  }
  const side = Math.round(Math.min(img.width, img.height) * 0.82);
  const sx = Math.round((img.width - side) / 2);
  const sy = Math.round(Math.max(0, (img.height - side) * 0.30));
  return { sx, sy, sw: side, sh: side };
}

// Affine triangle texture mapping onto canvas 2D path
function drawTexturedTriangle(ctx, img, p0, p1, p2, uv0, uv1, uv2, intensity, crop) {
  const x0 = p0.x, y0 = p0.y;
  const x1 = p1.x, y1 = p1.y;
  const x2 = p2.x, y2 = p2.y;

  const sx = crop ? crop.sx : 0;
  const sy = crop ? crop.sy : 0;
  const sw = crop ? crop.sw : img.width;
  const sh = crop ? crop.sh : img.height;

  const u0 = sx + uv0.u * sw;
  const v0 = sy + uv0.v * sh;
  const u1 = sx + uv1.u * sw;
  const v1 = sy + uv1.v * sh;
  const u2 = sx + uv2.u * sw;
  const v2 = sy + uv2.v * sh;

  const denom = u0 * (v1 - v2) + u1 * (v2 - v0) + u2 * (v0 - v1);
  if (Math.abs(denom) < 1e-4) return;

  const a = (x0 * (v1 - v2) + x1 * (v2 - v0) + x2 * (v0 - v1)) / denom;
  const b = (y0 * (v1 - v2) + y1 * (v2 - v0) + y2 * (v0 - v1)) / denom;
  const c = (x0 * (u2 - u1) + x1 * (u0 - u2) + x2 * (u1 - u0)) / denom;
  const d = (y0 * (u2 - u1) + y1 * (u0 - u2) + y2 * (u1 - u0)) / denom;
  const e = (x0 * (u1 * v2 - u2 * v1) + x1 * (u2 * v0 - u0 * v2) + x2 * (u0 * v1 - u1 * v0)) / denom;
  const f = (y0 * (u1 * v2 - u2 * v1) + y1 * (u2 * v0 - u0 * v2) + y2 * (u0 * v1 - u1 * v0)) / denom;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.closePath();
  ctx.clip();

  ctx.transform(a, b, c, d, e, f);
  ctx.drawImage(img, 0, 0);

  // Apply subtle 3D directional lighting/depth shadow
  if (intensity !== undefined && intensity < 0.98) {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.closePath();
    ctx.fillStyle = `rgba(0, 0, 0, ${Math.max(0, 1.0 - intensity) * 0.40})`;
    ctx.fill();
  }
  ctx.restore();
}

// Samples real hair and skin tone from subject's photo for seamless 360-degree anatomical shading
function extractFacePalette(img) {
  if (!img || !img.width || !img.height) {
    return { hair: [38, 34, 32], skin: [215, 172, 145] };
  }
  try {
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, 64, 64);
    // Hair sample near top center (x=32, y=5)
    const hairData = ctx.getImageData(30, 4, 5, 5).data;
    let hr = 0, hg = 0, hb = 0;
    for (let i = 0; i < hairData.length; i += 4) {
      hr += hairData[i];
      hg += hairData[i + 1];
      hb += hairData[i + 2];
    }
    const hCount = hairData.length / 4;
    const hair = [Math.round(hr / hCount), Math.round(hg / hCount), Math.round(hb / hCount)];

    // Skin sample near cheek (x=24, y=36)
    const skinData = ctx.getImageData(22, 34, 5, 5).data;
    let sr = 0, sg = 0, sb = 0;
    for (let i = 0; i < skinData.length; i += 4) {
      sr += skinData[i];
      sg += skinData[i + 1];
      sb += skinData[i + 2];
    }
    const sCount = skinData.length / 4;
    const skin = [Math.round(sr / sCount), Math.round(sg / sCount), Math.round(sb / sCount)];
    return { hair, skin };
  } catch {
    return { hair: [38, 34, 32], skin: [215, 172, 145] };
  }
}

export default function FaceMesh3DViewer({
  modelData,
  photoUrl = null,
  photos = [],
  title = "3D Face Model",
  badge = null,
  externalRotation = null,
  onRotate = null,
  mode = "photo", // 'photo' | 'shaded' | 'wireframe' | 'confidence' | 'comparison'
  deviations = null,
  width = 400,
  height = 380,
  showControls = true,
  interactive = true,
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [internalYaw, setInternalYaw] = useState(0);
  const [internalPitch, setInternalPitch] = useState(0);
  const [zoom, setZoom] = useState(1.0);
  const [autoRotate, setAutoRotate] = useState(false);
  const [renderMode, setRenderMode] = useState(mode);

  // Loaded real face textures
  const [loadedImages, setLoadedImages] = useState({
    front: null,
    left: null,
    right: null,
  });

  const facePalette = useMemo(() => {
    return extractFacePalette(loadedImages.front || loadedImages.left || loadedImages.right);
  }, [loadedImages]);

  const isDraggingRef = useRef(false);
  const lastMousePosRef = useRef({ x: 0, y: 0 });
  const animFrameRef = useRef(null);

  // Sync external rotation if provided
  const yaw = externalRotation ? externalRotation.yaw : internalYaw;
  const pitch = externalRotation ? externalRotation.pitch : internalPitch;

  useEffect(() => {
    setRenderMode(mode);
  }, [mode]);

  // Load photos into HTML Image objects for true-to-life 3D face texturing
  useEffect(() => {
    const urlsToLoad = {};
    if (photos && photos.length > 0) {
      photos.forEach((p) => {
        if (p.angle === "front" && p.previewUrl) urlsToLoad.front = p.previewUrl;
        if (p.angle === "left" && p.previewUrl) urlsToLoad.left = p.previewUrl;
        if (p.angle === "right" && p.previewUrl) urlsToLoad.right = p.previewUrl;
      });
      if (!urlsToLoad.front && photos[0]?.previewUrl) {
        urlsToLoad.front = photos[0].previewUrl;
      }
    } else if (photoUrl) {
      urlsToLoad.front = photoUrl;
    }

    const loaded = { front: null, left: null, right: null };
    let active = true;

    Object.entries(urlsToLoad).forEach(([key, url]) => {
      if (!url) return;
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = url;
      img.onload = () => {
        if (!active) return;
        setLoadedImages((prev) => ({ ...prev, [key]: img }));
      };
    });

    return () => {
      active = false;
    };
  }, [photoUrl, photos]);

  // Non-passive wheel event listener to fix "Unable to preventDefault inside passive event listener"
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onWheel = (e) => {
      if (!interactive) return;
      e.preventDefault();
      const delta = e.deltaY * -0.0015;
      setZoom((z) => Math.max(0.6, Math.min(2.4, z + delta)));
    };

    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [interactive]);

  // Auto-spin animation
  useEffect(() => {
    if (!autoRotate) return;
    let lastTime = performance.now();
    const spinLoop = (now) => {
      const dt = (now - lastTime) / 1000;
      lastTime = now;
      const newYaw = (yaw + dt * 0.45) % (Math.PI * 2);
      if (onRotate) {
        onRotate({ yaw: newYaw, pitch });
      } else {
        setInternalYaw(newYaw);
      }
      animFrameRef.current = requestAnimationFrame(spinLoop);
    };
    animFrameRef.current = requestAnimationFrame(spinLoop);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [autoRotate, yaw, pitch, onRotate]);

  // Resolve continuous 3D anatomical face mesh
  const mesh = useMemo(() => {
    if (modelData?.vertices && modelData.vertices.length >= 700 && modelData?.triangles) {
      const uvs = (modelData.uvs && modelData.uvs.length === modelData.vertices.length)
        ? modelData.uvs
        : modelData.vertices.map(([x, y]) => ({
            u: Math.max(0.01, Math.min(0.99, 0.50 + x * 0.58)),
            v: Math.max(0.01, Math.min(0.99, 0.55 - 0.75 * y)),
          }));
      return {
        vertices: modelData.vertices,
        triangles: modelData.triangles,
        uvs,
        confidences: modelData.confidence_per_vertex || [],
        isCranial: modelData.is_cranial || modelData.vertices.map(([x, y, z]) => z < 0.05 || Math.abs(x) > 0.44),
      };
    }
    const covered = photos && photos.length > 0 ? photos.map((p) => p.angle) : ["front"];
    return buildAnatomicalFaceMesh(modelData?.face_metadata, covered);
  }, [modelData, photos]);

  // Main 3D Rendering Engine
  const render3D = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || width;
    const h = canvas.clientHeight || height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    // Deep modern studio background with vignette
    const bgGrad = ctx.createRadialGradient(w / 2, h / 2, 20, w / 2, h / 2, Math.max(w, h) * 0.85);
    bgGrad.addColorStop(0, "#162030");
    bgGrad.addColorStop(0.65, "#0b1018");
    bgGrad.addColorStop(1, "#06090e");
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, w, h);

    // Subtle 3D volumetric depth cage rings
    ctx.strokeStyle = "rgba(124, 158, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let r = 50; r < Math.min(w, h) / 1.6; r += 50) {
      ctx.beginPath();
      ctx.ellipse(w / 2, h / 2, r, r * 0.85, 0, 0, Math.PI * 2);
      ctx.stroke();
    }

    const vertices = mesh.vertices;
    const triangles = mesh.triangles;
    const uvs = mesh.uvs;
    const confidences = mesh.confidences;

  // Precalculate trigonometric transforms
  const cosY = Math.cos(yaw);
  const sinY = Math.sin(yaw);
  const cosP = Math.cos(pitch);
  const sinP = Math.sin(pitch);

  const fov = 3.25 * zoom;
  const scale = Math.min(w, h) * 0.52;

  // Transform 3D vertices to 2D screen coordinates with perspective projection
  const transformed = vertices.map((v, idx) => {
    const [x, y, z] = v;
    // 1. Yaw (Y-axis)
    const x1 = x * cosY + z * sinY;
    const y1 = y;
    const z1 = -x * sinY + z * cosY;

    // 2. Pitch (X-axis)
    const x2 = x1;
    const y2 = y1 * cosP - z1 * sinP;
    const z2 = y1 * sinP + z1 * cosP;

    // 3. Perspective
    const cameraDist = 2.25;
    const depth = cameraDist - z2;
    const pScale = depth > 0.1 ? fov / depth : 0.01;

    const screenX = w / 2 + x2 * scale * pScale;
    const screenY = h / 2 - y2 * scale * pScale;

    const conf = confidences[idx] !== undefined ? confidences[idx] : 1.0;
    const dev = deviations ? deviations[idx] || 0 : 0;
    const uv = uvs[idx] || { u: 0.5, v: 0.5 };

    const cranialFlag = mesh.isCranial ? mesh.isCranial[idx] : (z < 0.05 || Math.abs(x) > 0.44);
    return { x: screenX, y: screenY, z: z2, origX: x, origY: y, origZ: z, conf, dev, uv, idx, isCranial: cranialFlag };
  });

  // Light source vector coming from front-top-right
  const lx = 0.45;
  const ly = 0.65;
  const lz = 0.85;
  const lLen = Math.hypot(lx, ly, lz);
  const lnx = lx / lLen;
  const lny = ly / lLen;
  const lnz = lz / lLen;

  // Determine active texture photo based on yaw rotation
  let activeTexture = loadedImages.front;
  if (yaw < -0.3 && loadedImages.left) {
    activeTexture = loadedImages.left;
  } else if (yaw > 0.3 && loadedImages.right) {
    activeTexture = loadedImages.right;
  }

  const crop = activeTexture ? getFaceCropRect(activeTexture, modelData?.face_metadata) : null;

  // ── 2. SORT TRIANGLES BY DEPTH (Painter's Algorithm) ──
  const sortedTris = triangles
    .map((tri, triIdx) => {
      const v0 = transformed[tri[0]];
      const v1 = transformed[tri[1]];
      const v2 = transformed[tri[2]];
      if (!v0 || !v1 || !v2) return null;
      const avgZ = (v0.z + v1.z + v2.z) / 3;
      const avgConf = (v0.conf + v1.conf + v2.conf) / 3;
      const avgDev = (v0.dev + v1.dev + v2.dev) / 3;
      return { tri, v0, v1, v2, avgZ, avgConf, avgDev, triIdx };
    })
    .filter(Boolean)
    .sort((a, b) => a.avgZ - b.avgZ);

  // ── 3. RENDER 3D FACETS ACCORDING TO ACTIVE MODE ──
  sortedTris.forEach((item) => {
    const { v0, v1, v2, avgConf, avgDev } = item;

    // Surface normal calculation for 3D Lambertian directional lighting
    const ax = v1.origX - v0.origX;
    const ay = v1.origY - v0.origY;
    const az = v1.origZ - v0.origZ;
    const bx = v2.origX - v0.origX;
    const by = v2.origY - v0.origY;
    const bz = v2.origZ - v0.origZ;

    let nx = ay * bz - az * by;
    let ny = az * bx - ax * bz;
    let nz = ax * by - ay * bx;
    const nLen = Math.hypot(nx, ny, nz) || 1;
    nx /= nLen;
    ny /= nLen;
    nz /= nLen;

    // Rotate surface normal with yaw and pitch
    const rnx = nx * cosY + nz * sinY;
    const rny = ny * cosP - (-nx * sinY + nz * cosY) * sinP;
    const rnz = ny * sinP + (-nx * sinY + nz * cosY) * cosP;

    // Backface culling: polygons facing away from camera are culled to prevent inside-out projection
    if (rnz <= 0.01) return;

    // Check cranial skull volume and lateral cheek facets
    const isPureCranial = (v0.isCranial ? 1 : 0) + (v1.isCranial ? 1 : 0) + (v2.isCranial ? 1 : 0) >= 2;
    const isLateral = Math.abs(v0.origX) > 0.38 || Math.abs(v1.origX) > 0.38 || Math.abs(v2.origX) > 0.38;

    // Continuous facial surface shading (Lambertian directional lighting)
    const dot = rnx * lnx + rny * lny + rnz * lnz;
    const intensity = Math.max(0.28, Math.min(1.0, 0.38 + 0.62 * Math.max(0, dot)));

    // Cranial facets: Render anatomical hair/skull shading with 3D directional light
    if (isPureCranial) {
      const [phR, phG, phB] = facePalette.hair;
      const hr = Math.floor(phR * intensity);
      const hg = Math.floor(phG * intensity);
      const hb = Math.floor(phB * intensity);
      ctx.beginPath();
      ctx.moveTo(v0.x, v0.y);
      ctx.lineTo(v1.x, v1.y);
      ctx.lineTo(v2.x, v2.y);
      ctx.closePath();
      ctx.fillStyle = `rgb(${hr}, ${hg}, ${hb})`;
      ctx.fill();
      if (renderMode === "wireframe") {
        ctx.strokeStyle = "rgba(100, 190, 255, 0.35)";
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
      return;
    }

    const uv0 = v0.uv;
    const uv1 = v1.uv;
    const uv2 = v2.uv;

    if (renderMode === "photo") {
      if (activeTexture) {
        drawTexturedTriangle(ctx, activeTexture, v0, v1, v2, uv0, uv1, uv2, intensity, crop);
        // If lateral facet without dedicated side profile photo, soften with sampled skin tone overlay
        if (isLateral && !loadedImages.left && !loadedImages.right) {
          const [psR, psG, psB] = facePalette.skin;
          ctx.beginPath();
          ctx.moveTo(v0.x, v0.y);
          ctx.lineTo(v1.x, v1.y);
          ctx.lineTo(v2.x, v2.y);
          ctx.closePath();
          ctx.fillStyle = `rgba(${psR}, ${psG}, ${psB}, 0.45)`;
          ctx.fill();
        }
      } else {
        // Realistic skin tone fallback if photo still loading
        const [psR, psG, psB] = facePalette.skin;
        const r = Math.floor(psR * intensity);
        const g = Math.floor(psG * intensity);
        const b = Math.floor(psB * intensity);
        ctx.beginPath();
        ctx.moveTo(v0.x, v0.y);
        ctx.lineTo(v1.x, v1.y);
        ctx.lineTo(v2.x, v2.y);
        ctx.closePath();
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.95)`;
        ctx.fill();
      }
    } else if (renderMode === "shaded") {
      const [psR, psG, psB] = facePalette.skin;
      const r = Math.floor(psR * intensity);
      const g = Math.floor(psG * intensity);
      const b = Math.floor(psB * intensity);
      ctx.beginPath();
      ctx.moveTo(v0.x, v0.y);
      ctx.lineTo(v1.x, v1.y);
      ctx.lineTo(v2.x, v2.y);
      ctx.closePath();
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.95)`;
      ctx.fill();
      ctx.strokeStyle = "rgba(120, 85, 70, 0.12)";
      ctx.lineWidth = 0.5;
      ctx.stroke();
    } else if (renderMode === "wireframe") {
      if (activeTexture) {
        drawTexturedTriangle(ctx, activeTexture, v0, v1, v2, uv0, uv1, uv2, intensity * 0.40, crop);
      }
      ctx.beginPath();
      ctx.moveTo(v0.x, v0.y);
      ctx.lineTo(v1.x, v1.y);
      ctx.lineTo(v2.x, v2.y);
      ctx.closePath();
      ctx.fillStyle = "rgba(14, 24, 40, 0.35)";
      ctx.fill();
      ctx.strokeStyle = "rgba(100, 190, 255, 0.70)";
      ctx.lineWidth = 0.7;
      ctx.stroke();
    } else if (renderMode === "confidence") {
      if (activeTexture) {
        drawTexturedTriangle(ctx, activeTexture, v0, v1, v2, uv0, uv1, uv2, intensity * 0.70, crop);
      }
      ctx.beginPath();
      ctx.moveTo(v0.x, v0.y);
      ctx.lineTo(v1.x, v1.y);
      ctx.lineTo(v2.x, v2.y);
      ctx.closePath();
      if (avgConf >= 0.85) {
        ctx.fillStyle = "rgba(34, 197, 94, 0.40)";
        ctx.strokeStyle = "rgba(34, 197, 94, 0.75)";
      } else if (avgConf >= 0.50) {
        ctx.fillStyle = "rgba(245, 158, 11, 0.50)";
        ctx.strokeStyle = "rgba(245, 158, 11, 0.85)";
      } else {
        ctx.fillStyle = "rgba(239, 68, 68, 0.60)";
        ctx.strokeStyle = "rgba(220, 38, 38, 0.95)";
      }
      ctx.lineWidth = 0.8;
      ctx.fill();
      ctx.stroke();
    } else if (renderMode === "comparison") {
      if (activeTexture) {
        drawTexturedTriangle(ctx, activeTexture, v0, v1, v2, uv0, uv1, uv2, intensity * 0.70, crop);
      }
      ctx.beginPath();
      ctx.moveTo(v0.x, v0.y);
      ctx.lineTo(v1.x, v1.y);
      ctx.lineTo(v2.x, v2.y);
      ctx.closePath();
      if (avgDev < 0.035) {
        ctx.fillStyle = "rgba(34, 197, 94, 0.40)";
        ctx.strokeStyle = "rgba(34, 197, 94, 0.75)";
      } else if (avgDev < 0.075) {
        ctx.fillStyle = "rgba(234, 179, 8, 0.50)";
        ctx.strokeStyle = "rgba(234, 179, 8, 0.85)";
      } else {
        ctx.fillStyle = "rgba(239, 68, 68, 0.60)";
        ctx.strokeStyle = "rgba(239, 68, 68, 0.95)";
      }
      ctx.lineWidth = 0.8;
      ctx.fill();
      ctx.stroke();
    }
  });

  // ── 4. KEY ANATOMICAL LANDMARKS ──
  // Show key landmarks in wireframe, confidence, and comparison modes to orient facial features
  if (renderMode !== "photo") {
    const keyLandmarks = [
      transformed.find((p) => Math.abs(p.origX - (-0.28)) < 0.07 && Math.abs(p.origY - 0.18) < 0.07),
      transformed.find((p) => Math.abs(p.origX - 0.28) < 0.07 && Math.abs(p.origY - 0.18) < 0.07),
      transformed.find((p) => Math.abs(p.origX) < 0.06 && Math.abs(p.origY - (-0.08)) < 0.07),
      transformed.find((p) => Math.abs(p.origX) < 0.06 && Math.abs(p.origY - (-0.30)) < 0.07),
      transformed.find((p) => Math.abs(p.origX) < 0.06 && Math.abs(p.origY - (-0.38)) < 0.07),
      transformed.find((p) => Math.abs(p.origX) < 0.06 && Math.abs(p.origY - (-0.58)) < 0.07),
      transformed.find((p) => Math.abs(p.origX - (-0.38)) < 0.07 && Math.abs(p.origY) < 0.08),
      transformed.find((p) => Math.abs(p.origX - 0.38) < 0.07 && Math.abs(p.origY) < 0.08),
    ].filter(Boolean);

    keyLandmarks.forEach((pt) => {
      const nodeRad = Math.max(2.2, Math.min(4.2, (pt.z + 1.0) * 2.2));
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, nodeRad, 0, Math.PI * 2);

      if (renderMode === "confidence") {
        ctx.fillStyle = pt.conf >= 0.85 ? "#22c55e" : pt.conf >= 0.5 ? "#f59e0b" : "#ef4444";
      } else if (renderMode === "comparison") {
        ctx.fillStyle = pt.dev < 0.035 ? "#4ade80" : pt.dev < 0.075 ? "#facc15" : "#f87171";
      } else if (renderMode === "shaded") {
        ctx.fillStyle = "#e2b89b";
      } else {
        ctx.fillStyle = "rgba(147, 197, 253, 0.95)";
      }
      ctx.fill();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  }

    // ── 5. 3D COORDINATE AXES INDICATOR ──
    const axisX = 36;
    const axisY = h - 36;
    const axisLen = 20;
    // X (Red)
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(axisX, axisY);
    ctx.lineTo(axisX + cosY * axisLen, axisY);
    ctx.stroke();
    // Y (Green)
    ctx.strokeStyle = "#22c55e";
    ctx.beginPath();
    ctx.moveTo(axisX, axisY);
    ctx.lineTo(axisX, axisY - cosP * axisLen);
    ctx.stroke();
    // Z (Blue)
    ctx.strokeStyle = "#3b82f6";
    ctx.beginPath();
    ctx.moveTo(axisX, axisY);
    ctx.lineTo(axisX + sinY * axisLen * 0.7, axisY + sinP * axisLen * 0.7);
  }, [mesh, modelData, yaw, pitch, zoom, renderMode, deviations, loadedImages, width, height]);

  useEffect(() => {
    render3D();
  }, [render3D]);

  // Mouse & Touch Drag Handlers
  const handleMouseDown = (e) => {
    if (!interactive) return;
    isDraggingRef.current = true;
    lastMousePosRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e) => {
    if (!isDraggingRef.current || !interactive) return;
    const dx = e.clientX - lastMousePosRef.current.x;
    const dy = e.clientY - lastMousePosRef.current.y;
    lastMousePosRef.current = { x: e.clientX, y: e.clientY };

    const sensitivity = 0.012;
    const nextYaw = yaw + dx * sensitivity;
    const nextPitch = Math.max(-1.3, Math.min(1.3, pitch + dy * sensitivity));

    if (onRotate) {
      onRotate({ yaw: nextYaw, pitch: nextPitch });
    } else {
      setInternalYaw(nextYaw);
      setInternalPitch(nextPitch);
    }
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  const handleTouchStart = (e) => {
    if (!interactive || e.touches.length !== 1) return;
    isDraggingRef.current = true;
    lastMousePosRef.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
  };

  const handleTouchMove = (e) => {
    if (!isDraggingRef.current || !interactive || e.touches.length !== 1) return;
    const dx = e.touches[0].clientX - lastMousePosRef.current.x;
    const dy = e.touches[0].clientY - lastMousePosRef.current.y;
    lastMousePosRef.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };

    const sensitivity = 0.012;
    const nextYaw = yaw + dx * sensitivity;
    const nextPitch = Math.max(-1.3, Math.min(1.3, pitch + dy * sensitivity));

    if (onRotate) {
      onRotate({ yaw: nextYaw, pitch: nextPitch });
    } else {
      setInternalYaw(nextYaw);
      setInternalPitch(nextPitch);
    }
  };

  const setViewPreset = (yDeg, pDeg) => {
    const nextY = (yDeg * Math.PI) / 180;
    const nextP = (pDeg * Math.PI) / 180;
    if (onRotate) {
      onRotate({ yaw: nextY, pitch: nextP });
    } else {
      setInternalYaw(nextY);
      setInternalPitch(nextP);
    }
  };

  const completeness = modelData?.completeness_score ?? 100;
  const isComplete = completeness >= 95;

  return (
    <div className="viewer-3d-wrapper" ref={containerRef}>
      {/* ── Card Header ── */}
      <div className="viewer-3d-header">
        <div className="viewer-3d-title-row">
          <span className="viewer-title">{title}</span>
          {badge && <span className="viewer-badge-tag">{badge}</span>}
          <span
            className={`completeness-pill ${isComplete ? "pill-complete" : "pill-incomplete"}`}
            title={isComplete ? "All 5 reference angles present" : "Missing viewpoints detected"}
          >
            {isComplete ? "✓ 100% 3D Constrained" : `⚠️ ${completeness}% Completeness`}
          </span>
        </div>

        {/* View Mode Switcher */}
        {showControls && (
          <div className="viewer-mode-pills">
            <button
              className={`mode-pill ${renderMode === "photo" ? "active" : ""}`}
              onClick={() => setRenderMode("photo")}
              title="True-to-life 3D Photo Projection"
            >
              Photo 3D
            </button>
            <button
              className={`mode-pill ${renderMode === "shaded" ? "active" : ""}`}
              onClick={() => setRenderMode("shaded")}
              title="Realistic Anatomical Skin Surface"
            >
              Surface
            </button>
            <button
              className={`mode-pill ${renderMode === "wireframe" ? "active" : ""}`}
              onClick={() => setRenderMode("wireframe")}
              title="Polygonal Lattice Mesh"
            >
              Wireframe
            </button>
            <button
              className={`mode-pill ${renderMode === "confidence" ? "active" : ""}`}
              onClick={() => setRenderMode("confidence")}
              title="Inconsistency & Uncertainty Heatmap"
            >
              Inconsistencies
            </button>
            {deviations && (
              <button
                className={`mode-pill ${renderMode === "comparison" ? "active" : ""}`}
                onClick={() => setRenderMode("comparison")}
                title="3D Structural Geometric Discrepancy"
              >
                Deviation
              </button>
            )}
          </div>
        )}
      </div>

      {/* ── 3D Viewport ── */}
      <div
        className="viewer-3d-canvas-container"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleMouseUp}
      >
        <canvas
          ref={canvasRef}
          style={{ width: "100%", height: `${height}px`, display: "block", cursor: "grab" }}
        />

        {/* Floating HUD Overlay */}
        <div className="viewer-hud-overlay">
          <span>Yaw: {Math.round((yaw * 180) / Math.PI)}°</span>
          <span>Pitch: {Math.round((pitch * 180) / Math.PI)}°</span>
          <span>Zoom: {zoom.toFixed(1)}x</span>
        </div>

        {/* Inconsistency Warning Overlay Banner */}
        {renderMode === "confidence" && !isComplete && (
          <div className="viewer-inconsistency-badge">
            <span className="dot pulse-red" />
            <span>Red/Amber: Missing Views & Depth Ambiguities</span>
          </div>
        )}

        {/* Inconsistency Heatmap Legend */}
        {renderMode === "confidence" && (
          <div className="viewer-legend-card">
            <div className="legend-row">
              <span className="legend-swatch swatch-green" />
              <span>Multi-View Confirmed (&ge;90%)</span>
            </div>
            <div className="legend-row">
              <span className="legend-swatch swatch-amber" />
              <span>Partial / Interpolated (50-89%)</span>
            </div>
            <div className="legend-row">
              <span className="legend-swatch swatch-red" />
              <span>Blind Spot / Inconsistent (&lt;50%)</span>
            </div>
          </div>
        )}

        {/* Comparison Legend */}
        {renderMode === "comparison" && (
          <div className="viewer-legend-card">
            <div className="legend-row">
              <span className="legend-swatch swatch-green" />
              <span>Matching Shape (&lt;2mm)</span>
            </div>
            <div className="legend-row">
              <span className="legend-swatch swatch-amber" />
              <span>Mild Variation (2-5mm)</span>
            </div>
            <div className="legend-row">
              <span className="legend-swatch swatch-red" />
              <span>Structural Divergence (&gt;5mm)</span>
            </div>
          </div>
        )}
      </div>

      {/* ── Controls Toolbar ── */}
      {showControls && (
        <div className="viewer-3d-toolbar">
          <div className="preset-buttons">
            <button
              className="ctrl-btn"
              onClick={() => setViewPreset(0, 0)}
              title="Frontal View (0°)"
            >
              Front
            </button>
            <button
              className="ctrl-btn"
              onClick={() => setViewPreset(-35, 5)}
              title="Left Profile 35°"
            >
              Left 35°
            </button>
            <button
              className="ctrl-btn"
              onClick={() => setViewPreset(35, 5)}
              title="Right Profile 35°"
            >
              Right 35°
            </button>
            <button
              className="ctrl-btn"
              onClick={() => setViewPreset(0, -25)}
              title="Chin Tilt Up 25°"
            >
              Tilt Up
            </button>
            <button
              className="ctrl-btn"
              onClick={() => setViewPreset(0, 25)}
              title="Forehead Tilt Down 25°"
            >
              Tilt Down
            </button>
          </div>

          <div className="tool-actions">
            <button
              className={`ctrl-btn spin-btn ${autoRotate ? "active-spin" : ""}`}
              onClick={() => setAutoRotate(!autoRotate)}
              title="Toggle Continuous 3D Spin"
            >
              {autoRotate ? "⏹ Stop Spin" : "↻ Auto-Spin"}
            </button>
            <button
              className="ctrl-btn"
              onClick={() => {
                setViewPreset(0, 0);
                setZoom(1.0);
              }}
              title="Reset View and Zoom"
            >
              ↺ Reset
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
