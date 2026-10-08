import { useEffect, useRef, useState, useCallback } from "react";

// 52 anatomically grounded landmarks in normalized 3D coordinates [-1, 1]
const CANONICAL_LANDMARKS = [
  // Forehead & hairline (0-5)
  [-0.0, 0.72, -0.05], [-0.36, 0.68, -0.16], [0.36, 0.68, -0.16],
  [0.0, 0.52, 0.12], [-0.32, 0.50, 0.04], [0.32, 0.50, 0.04],
  // Temples & Glabella (6-8)
  [-0.60, 0.42, -0.22], [0.60, 0.42, -0.22], [0.0, 0.34, 0.24],
  // Brows (9-14)
  [-0.16, 0.34, 0.23], [-0.36, 0.36, 0.19], [-0.52, 0.32, 0.06],
  [0.16, 0.34, 0.23], [0.36, 0.36, 0.19], [0.52, 0.32, 0.06],
  // Eyes & Sockets (15-24)
  [-0.15, 0.20, 0.15], [-0.30, 0.20, 0.13], [-0.46, 0.20, 0.07],
  [-0.30, 0.25, 0.17], [-0.30, 0.15, 0.11], [0.15, 0.20, 0.15],
  [0.30, 0.20, 0.13], [0.46, 0.20, 0.07], [0.30, 0.25, 0.17],
  [0.30, 0.15, 0.11],
  // Nose (25-33)
  [0.0, 0.22, 0.26], [0.0, 0.08, 0.35], [0.0, -0.04, 0.43],
  [0.0, -0.10, 0.48], [0.0, -0.20, 0.32], [-0.16, -0.11, 0.30],
  [-0.12, -0.19, 0.26], [0.16, -0.11, 0.30], [0.12, -0.19, 0.26],
  // Cheeks & Zygoma (34-39)
  [-0.32, 0.05, 0.09], [-0.58, 0.04, -0.09], [-0.42, -0.15, 0.05],
  [0.32, 0.05, 0.09], [0.58, 0.04, -0.09], [0.42, -0.15, 0.05],
  // Lips & Mouth (40-47)
  [0.0, -0.25, 0.30], [0.0, -0.30, 0.31], [-0.09, -0.29, 0.30],
  [0.09, -0.29, 0.30], [0.0, -0.35, 0.27], [0.0, -0.41, 0.29],
  [-0.24, -0.35, 0.19], [0.24, -0.35, 0.19],
  // Chin & Jawline (48-56)
  [0.0, -0.49, 0.24], [0.0, -0.59, 0.28], [0.0, -0.70, 0.18],
  [-0.18, -0.63, 0.20], [0.18, -0.63, 0.20], [-0.54, -0.42, -0.18],
  [0.54, -0.42, -0.18], [-0.38, -0.55, -0.02], [0.38, -0.55, -0.02]
];

// Polygonal triangle facets
const CANONICAL_TRIANGLES = [
  // Forehead
  [0, 1, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
  [1, 6, 4], [2, 5, 7],
  // Brow & Glabella
  [3, 4, 9], [3, 9, 8], [3, 8, 12], [3, 12, 5],
  [4, 6, 11], [4, 11, 10], [4, 10, 9],
  [5, 12, 13], [5, 13, 14], [5, 14, 7],
  // Nose bridge
  [8, 9, 25], [8, 25, 12],
  [25, 15, 26], [25, 26, 20],
  [26, 15, 34], [26, 37, 20],
  // Eyes
  [9, 10, 18], [9, 18, 15], [10, 11, 17], [10, 17, 18], [15, 18, 16], [18, 17, 16],
  [15, 16, 19], [16, 17, 19], [17, 6, 35], [17, 35, 19],
  [12, 23, 20], [12, 13, 23], [13, 24, 23], [13, 14, 22], [13, 22, 24],
  [20, 21, 24], [21, 22, 24], [22, 7, 38], [22, 38, 24],
  // Nose body & tip
  [26, 34, 30], [26, 30, 27], [26, 27, 32], [26, 32, 37],
  [27, 30, 28], [27, 28, 32],
  [28, 30, 31], [28, 31, 29], [28, 29, 33], [28, 33, 32],
  // Cheeks
  [19, 35, 36], [19, 36, 34], [34, 36, 30],
  [24, 39, 38], [24, 37, 39], [37, 32, 39],
  // Mouth & Philtrum
  [29, 31, 40], [29, 40, 33],
  [40, 31, 42], [40, 42, 41], [40, 41, 43], [40, 43, 33],
  [41, 42, 44], [41, 44, 43],
  [42, 46, 44], [43, 44, 47],
  [30, 36, 46], [30, 46, 31], [31, 46, 42],
  [32, 33, 47], [32, 47, 39], [39, 47, 36],
  // Lips & Chin
  [44, 46, 45], [44, 45, 47],
  [45, 46, 48], [45, 48, 47],
  [48, 46, 51], [48, 51, 49], [48, 49, 52], [48, 52, 47],
  [49, 51, 50], [49, 50, 52],
  // Jawline & Mandible
  [36, 53, 55], [36, 55, 46], [46, 55, 51], [51, 55, 50],
  [39, 47, 56], [39, 56, 54], [47, 52, 56], [52, 50, 56],
  [35, 53, 36], [38, 39, 54],
];

// Facial perimeter contour indices
const CONTOUR_INDICES = [0, 1, 6, 35, 53, 55, 51, 50, 52, 56, 54, 38, 7, 2, 0];

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

    const vertices = modelData?.vertices || CANONICAL_LANDMARKS;
    const triangles = modelData?.triangles || CANONICAL_TRIANGLES;
    const confidences = modelData?.confidence_per_vertex || [];

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

      return { x: screenX, y: screenY, z: z2, origX: x, origY: y, origZ: z, conf, dev, idx };
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

    // ── 1. RENDER PHOTO-TEXTURED 3D FACE SURFACE ──
    if (activeTexture && (renderMode === "photo" || renderMode === "wireframe" || renderMode === "confidence" || renderMode === "comparison")) {
      ctx.save();

      // Create 3D facial clipping path along the perimeter contour
      ctx.beginPath();
      CONTOUR_INDICES.forEach((idx, i) => {
        const pt = transformed[idx];
        if (!pt) return;
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      });
      ctx.closePath();
      ctx.clip();

      // Calculate perspective image placement
      // Center on nose bridge landmark [26] or [8]
      const anchorPt = transformed[26] || transformed[8] || { x: w / 2, y: h / 2 };
      const chinPt = transformed[50] || { x: w / 2, y: h * 0.85 };
      const foreheadPt = transformed[0] || { x: w / 2, y: h * 0.2 };
      const faceHeight = Math.hypot(chinPt.x - foreheadPt.x, chinPt.y - foreheadPt.y) || h * 0.7;

      const imgAspect = activeTexture.width / activeTexture.height;
      const drawH = faceHeight * 1.55;
      const drawW = drawH * imgAspect;

      // Parallax shift based on yaw & pitch
      const shiftX = Math.sin(yaw) * (w * 0.14);
      const shiftY = -Math.sin(pitch) * (h * 0.12);

      ctx.drawImage(
        activeTexture,
        anchorPt.x - drawW / 2 + shiftX,
        anchorPt.y - drawH * 0.46 + shiftY,
        drawW,
        drawH
      );

      // Apply 3D anatomical shading & depth illumination over the photo
      const lightGrad = ctx.createLinearGradient(
        w / 2 - Math.sin(yaw) * 120,
        h / 2 - Math.sin(pitch) * 120,
        w / 2 + Math.sin(yaw) * 120,
        h / 2 + Math.sin(pitch) * 120
      );
      // Dynamic rim lighting & depth falloff
      const depthShadow = Math.abs(Math.sin(yaw)) * 0.35;
      lightGrad.addColorStop(0, `rgba(255, 255, 255, ${0.15 - depthShadow * 0.1})`);
      lightGrad.addColorStop(0.5, "rgba(0, 0, 0, 0)");
      lightGrad.addColorStop(1, `rgba(0, 0, 0, ${0.35 + depthShadow})`);

      ctx.fillStyle = lightGrad;
      ctx.fillRect(0, 0, w, h);

      ctx.restore();
    }

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

      // Rotate normal
      const rnx = nx * cosY + nz * sinY;
      const rny = ny * cosP - (-nx * sinY + nz * cosY) * sinP;
      const rnz = ny * sinP + (-nx * sinY + nz * cosY) * cosP;

      const dot = Math.max(0, rnx * lnx + rny * lny + rnz * lnz);
      const intensity = 0.3 + 0.7 * dot;

      ctx.beginPath();
      ctx.moveTo(v0.x, v0.y);
      ctx.lineTo(v1.x, v1.y);
      ctx.lineTo(v2.x, v2.y);
      ctx.closePath();

      if (renderMode === "photo") {
        // Subtle 3D depth facet relief over the photo
        ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        ctx.lineWidth = 0.6;
        ctx.stroke();
      } else if (renderMode === "shaded") {
        // Natural realistic skin tone shading (NOT a masquerade mask!)
        // Warm natural human facial tone with diffuse lighting
        const r = Math.floor(215 * intensity);
        const g = Math.floor(168 * intensity);
        const b = Math.floor(142 * intensity);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.92)`;
        ctx.fill();

        ctx.strokeStyle = "rgba(120, 85, 70, 0.18)";
        ctx.lineWidth = 0.7;
        ctx.stroke();
      } else if (renderMode === "wireframe") {
        // High-tech holographic wireframe lattice over the face
        ctx.fillStyle = "rgba(14, 24, 40, 0.35)";
        ctx.fill();

        ctx.strokeStyle = "rgba(100, 190, 255, 0.65)";
        ctx.lineWidth = 1.0;
        ctx.stroke();
      } else if (renderMode === "confidence") {
        // INCONSISTENCY & UNCERTAINTY HEATMAP
        // Green: confirmed, Amber: partial, Red: blind spot / inconsistent
        if (avgConf >= 0.85) {
          ctx.fillStyle = `rgba(34, 197, 94, ${0.28 + 0.35 * intensity})`;
          ctx.strokeStyle = "rgba(34, 197, 94, 0.85)";
          ctx.lineWidth = 0.8;
        } else if (avgConf >= 0.50) {
          ctx.fillStyle = `rgba(245, 158, 11, ${0.35 + 0.35 * intensity})`;
          ctx.strokeStyle = "rgba(245, 158, 11, 0.95)";
          ctx.lineWidth = 1.2;
        } else {
          // Unconstrained blind spot
          ctx.fillStyle = `rgba(239, 68, 68, ${0.45 + 0.4 * intensity})`;
          ctx.strokeStyle = "rgba(220, 38, 38, 1.0)";
          ctx.lineWidth = 1.8;
        }
        ctx.fill();
        ctx.stroke();
      } else if (renderMode === "comparison") {
        // 3D GEOMETRIC DEVIATION HEATMAP
        if (avgDev < 0.035) {
          ctx.fillStyle = `rgba(34, 197, 94, ${0.3 + 0.35 * intensity})`;
          ctx.strokeStyle = "rgba(34, 197, 94, 0.85)";
        } else if (avgDev < 0.075) {
          ctx.fillStyle = `rgba(234, 179, 8, ${0.38 + 0.35 * intensity})`;
          ctx.strokeStyle = "rgba(234, 179, 8, 0.9)";
        } else {
          ctx.fillStyle = `rgba(239, 68, 68, ${0.5 + 0.4 * intensity})`;
          ctx.strokeStyle = "rgba(239, 68, 68, 1.0)";
        }
        ctx.fill();
        ctx.stroke();
      }
    });

    // ── 4. GLOWING 3D ANATOMICAL LANDMARKS ──
    if (renderMode !== "photo" || Math.abs(yaw) > 0.05) {
      transformed.forEach((pt) => {
        const nodeRad = Math.max(1.8, Math.min(3.8, (pt.z + 1.0) * 2.0));
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, nodeRad, 0, Math.PI * 2);

        if (renderMode === "confidence") {
          ctx.fillStyle = pt.conf >= 0.85 ? "#22c55e" : pt.conf >= 0.5 ? "#f59e0b" : "#ef4444";
        } else if (renderMode === "comparison") {
          ctx.fillStyle = pt.dev < 0.035 ? "#4ade80" : pt.dev < 0.075 ? "#facc15" : "#f87171";
        } else if (renderMode === "shaded") {
          ctx.fillStyle = "#e2b89b";
        } else {
          ctx.fillStyle = "rgba(147, 197, 253, 0.85)";
        }
        ctx.fill();
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
    ctx.stroke();
  }, [modelData, yaw, pitch, zoom, renderMode, deviations, loadedImages, width, height]);

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
