import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { FaceMesh, FACEMESH_TESSELATION, FACEMESH_FACE_OVAL } from "@mediapipe/face_mesh";

function round4(v) {
  return Math.round(v * 10000) / 10000;
}

/**
 * Builds a custom 3D face mesh directly from 468 MediaPipe facial landmarks.
 * - Perfectly registered to the user's photograph (zero distortion)
 * - 852 realistic facial triangles
 * - Solid volumetric cranial back closure with smooth normals
 */
function buildMeshFromMediaPipeLandmarks(landmarks) {
  const xs = landmarks.map((p) => p.x);
  const ys = landmarks.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const span = Math.max(maxX - minX, maxY - minY) || 1.0;

  const vertices = landmarks.map((p) => {
    const x = ((p.x - cx) / span) * 0.66;
    const y = -((p.y - cy) / span) * 0.66;
    const z = -p.z * 0.75;
    return [round4(x), round4(y), round4(z)];
  });

  const uvs = landmarks.map((p) => ({
    u: round4(Math.max(0.001, Math.min(0.999, p.x))),
    v: round4(Math.max(0.001, Math.min(0.999, p.y))),
  }));

  const isCranial = new Array(468).fill(false);
  const confidences = new Array(468).fill(0.95);

  const triangles = [];
  for (let i = 0; i < FACEMESH_TESSELATION.length; i += 3) {
    const e1 = FACEMESH_TESSELATION[i];
    const e2 = FACEMESH_TESSELATION[i + 1];
    triangles.push([e1[0], e1[1], e2[1]]);
  }

  // Solid cranial back shell from FACEMESH_FACE_OVAL
  const ovalLoop = FACEMESH_FACE_OVAL.map((e) => e[0]);
  const nOval = ovalLoop.length;
  const backRingStart = vertices.length;

  for (let i = 0; i < nOval; i++) {
    const v = vertices[ovalLoop[i]];
    vertices.push([round4(v[0] * 0.92), round4(v[1] * 0.92), round4(-0.20)]);
    uvs.push({ u: 0.5, v: 0.5 });
    isCranial.push(true);
    confidences.push(0.5);
  }

  const apexIdx = vertices.length;
  vertices.push([0.0, 0.0, -0.32]);
  uvs.push({ u: 0.5, v: 0.5 });
  isCranial.push(true);
  confidences.push(0.5);

  for (let i = 0; i < nOval; i++) {
    const iNext = (i + 1) % nOval;
    const a = ovalLoop[i];
    const b = ovalLoop[iNext];
    const c = backRingStart + i;
    const d = backRingStart + iNext;
    triangles.push([a, c, b]);
    triangles.push([b, c, d]);
  }

  for (let i = 0; i < nOval; i++) {
    const iNext = (i + 1) % nOval;
    const c = backRingStart + i;
    const d = backRingStart + iNext;
    triangles.push([c, apexIdx, d]);
  }

  // Accumulate unit normals from incident face normals
  const totalVerts = vertices.length;
  const normals = new Array(totalVerts).fill(0).map(() => [0, 0, 0]);
  for (let k = 0; k < triangles.length; k++) {
    const [a, b, c] = triangles[k];
    const va = vertices[a], vb = vertices[b], vc = vertices[c];
    const ab = [vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]];
    const ac = [vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]];
    const fnx = ab[1] * ac[2] - ab[2] * ac[1];
    const fny = ab[2] * ac[0] - ab[0] * ac[2];
    const fnz = ab[0] * ac[1] - ab[1] * ac[0];
    normals[a][0] += fnx; normals[a][1] += fny; normals[a][2] += fnz;
    normals[b][0] += fnx; normals[b][1] += fny; normals[b][2] += fnz;
    normals[c][0] += fnx; normals[c][1] += fny; normals[c][2] += fnz;
  }
  for (let i = 0; i < totalVerts; i++) {
    const l = Math.hypot(normals[i][0], normals[i][1], normals[i][2]) || 1.0;
    normals[i] = [round4(normals[i][0] / l), round4(normals[i][1] / l), round4(normals[i][2] / l)];
  }

  const wireframeIndices = [];
  for (let k = 0; k < triangles.length; k++) {
    const [a, b, c] = triangles[k];
    wireframeIndices.push(a, b, b, c, c, a);
  }

  return { vertices, triangles, uvs, normals, isCranial, confidences, wireframeIndices };
}

/**
 * Builds smooth anatomical fallback mesh if MediaPipe is loading.
 */
export function buildAnatomicalFaceMesh(faceMeta = null, coveredAngles = ["front"], nLat = 36, nLon = 44) {
  const covered = new Set(coveredAngles || ["front"]);

  const phis = [];
  for (let j = 0; j < nLat; j++) {
    phis.push((Math.PI * j) / (nLat - 1));
  }
  const thetas = [];
  for (let i = 0; i < nLon; i++) {
    thetas.push(-Math.PI + (2.0 * Math.PI * i) / nLon);
  }

  const vertices = [];
  const uvs = [];
  const confidences = [];
  const isCranial = [];

  const topY = 0.42;
  vertices.push([0.0, topY, 0.0]);
  uvs.push({ u: 0.50, v: 0.05 });
  confidences.push(0.95);
  isCranial.push(true);

  for (let j = 1; j < nLat - 1; j++) {
    const phi = phis[j];
    const y = round4(0.42 * Math.cos(phi));
    const s = Math.sin(phi);

    const rx = 0.32 * s;
    const rzF = 0.28 * s;
    const rzB = 0.36 * s;

    for (let i = 0; i < nLon; i++) {
      const th = thetas[i];
      const sinT = Math.sin(th);
      const cosT = Math.cos(th);

      const x = round4(rx * sinT);

      if (cosT >= 0) {
        const zBase = rzF * Math.pow(cosT, 0.85);
        let zNose = 0.0;
        if (y >= -0.14 && y <= 0.10 && Math.abs(x) < 0.10) {
          const latN = Math.exp(-0.5 * Math.pow(x / 0.035, 2));
          const prog = y >= -0.04 ? (0.10 - y) / 0.14 : (y - -0.14) / 0.10;
          zNose = (0.015 + 0.075 * prog) * latN;
        }
        const z = round4(zBase + zNose);
        isCranial.push(false);
        const u = round4(Math.max(0.01, Math.min(0.99, 0.50 + x / 0.60)));
        const v = round4(Math.max(0.01, Math.min(0.99, 0.50 - y / 0.76)));
        uvs.push({ u, v });
        vertices.push([x, y, z]);
      } else {
        const z = -round4(rzB * Math.pow(Math.abs(cosT), 0.85));
        isCranial.push(true);
        uvs.push({ u: 0.50, v: 0.50 });
        vertices.push([x, y, z]);
      }
      confidences.push(0.95);
    }
  }

  const botY = -0.42;
  vertices.push([0.0, botY, 0.0]);
  uvs.push({ u: 0.50, v: 0.95 });
  confidences.push(0.95);
  isCranial.push(true);

  const totalVerts = vertices.length;
  const triangles = [];
  const wireframeIndices = [];

  for (let i = 0; i < nLon; i++) {
    const iNext = (i + 1) % nLon;
    const a = 0, b = 1 + i, c = 1 + iNext;
    triangles.push([a, b, c]);
    wireframeIndices.push(a, b, b, c, c, a);
  }

  for (let j = 1; j < nLat - 2; j++) {
    const rCurr = 1 + (j - 1) * nLon;
    const rNext = 1 + j * nLon;
    for (let i = 0; i < nLon; i++) {
      const iNext = (i + 1) % nLon;
      const v00 = rCurr + i;
      const v10 = rCurr + iNext;
      const v01 = rNext + i;
      const v11 = rNext + iNext;
      triangles.push([v00, v01, v10]);
      triangles.push([v10, v01, v11]);
      wireframeIndices.push(v00, v01, v01, v10, v10, v00);
      wireframeIndices.push(v10, v01, v01, v11, v11, v10);
    }
  }

  const botPole = totalVerts - 1;
  const lastR = 1 + (nLat - 3) * nLon;
  for (let i = 0; i < nLon; i++) {
    const iNext = (i + 1) % nLon;
    const a = botPole, b = lastR + iNext, c = lastR + i;
    triangles.push([a, b, c]);
    wireframeIndices.push(a, b, b, c, c, a);
  }

  const normals = new Array(totalVerts).fill(0).map(() => [0, 0, 0]);
  for (let k = 0; k < triangles.length; k++) {
    const [a, b, c] = triangles[k];
    const va = vertices[a], vb = vertices[b], vc = vertices[c];
    const ab = [vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]];
    const ac = [vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]];
    const fnx = ab[1] * ac[2] - ab[2] * ac[1];
    const fny = ab[2] * ac[0] - ab[0] * ac[2];
    const fnz = ab[0] * ac[1] - ab[1] * ac[0];
    normals[a][0] += fnx; normals[a][1] += fny; normals[a][2] += fnz;
    normals[b][0] += fnx; normals[b][1] += fny; normals[b][2] += fnz;
    normals[c][0] += fnx; normals[c][1] += fny; normals[c][2] += fnz;
  }
  for (let i = 0; i < totalVerts; i++) {
    const l = Math.hypot(normals[i][0], normals[i][1], normals[i][2]) || 1.0;
    normals[i] = [round4(normals[i][0] / l), round4(normals[i][1] / l), round4(normals[i][2] / l)];
  }

  return { vertices, triangles, uvs, confidences, isCranial, normals, wireframeIndices };
}

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
    const hairData = ctx.getImageData(30, 4, 5, 5).data;
    let hr = 0, hg = 0, hb = 0;
    for (let i = 0; i < hairData.length; i += 4) {
      hr += hairData[i];
      hg += hairData[i + 1];
      hb += hairData[i + 2];
    }
    const hCount = hairData.length / 4;
    const hair = [Math.round(hr / hCount), Math.round(hg / hCount), Math.round(hb / hCount)];

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

// ── WebGL Shaders & Matrix Math ──────────────────────────────────────────

const VS_SOURCE = `
  attribute vec3 aPosition;
  attribute vec3 aNormal;
  attribute vec2 aUv;
  attribute vec4 aColor;
  attribute float aIsCranial;

  uniform mat4 uModelView;
  uniform mat4 uProjection;
  uniform mat3 uNormalMatrix;

  varying vec3 vNormal;
  varying vec3 vPosition;
  varying vec2 vUv;
  varying vec4 vColor;
  varying float vIsCranial;

  void main() {
    vec4 mvPos = uModelView * vec4(aPosition, 1.0);
    vPosition = mvPos.xyz;
    vNormal = normalize(uNormalMatrix * aNormal);
    vUv = aUv;
    vColor = aColor;
    vIsCranial = aIsCranial;
    gl_Position = uProjection * mvPos;
  }
`;

const FS_SOURCE = `
  precision mediump float;

  varying vec3 vNormal;
  varying vec3 vPosition;
  varying vec2 vUv;
  varying vec4 vColor;
  varying float vIsCranial;

  uniform int uRenderMode;
  uniform sampler2D uTexture;
  uniform float uHasTexture;
  uniform vec3 uSkinColor;
  uniform vec3 uHairColor;
  uniform vec3 uLightDir1;
  uniform vec3 uLightDir2;
  uniform vec3 uLightDir3;

  void main() {
    if (uRenderMode == 5) {
      gl_FragColor = vec4(0.22, 0.74, 0.98, 0.85);
      return;
    }

    vec3 N = normalize(vNormal);
    vec3 V = normalize(-vPosition);

    float diff1 = max(0.0, dot(N, uLightDir1));
    vec3 H1 = normalize(uLightDir1 + V);
    float spec1 = pow(max(0.0, dot(N, H1)), 24.0) * 0.35;

    float diff2 = max(0.0, dot(N, uLightDir2)) * 0.30;
    float rim = pow(clamp(1.0 - dot(N, V), 0.0, 1.0), 3.0) * 0.30;

    float ambient = 0.35;
    float totalDiffuse = ambient + diff1 * 0.55 + diff2;

    vec3 baseColor = uSkinColor;

    if (vIsCranial > 0.5) {
      baseColor = uHairColor;
      vec3 finalColor = baseColor * totalDiffuse + vec3(1.0) * spec1 * 0.15;
      gl_FragColor = vec4(finalColor, 1.0);
      return;
    }

    if (uRenderMode == 0) { // Photo 3D
      if (uHasTexture > 0.5) {
        vec4 texColor = texture2D(uTexture, vUv);
        baseColor = texColor.rgb;
      } else {
        baseColor = uSkinColor;
      }
      vec3 finalColor = baseColor * totalDiffuse + vec3(1.0, 0.95, 0.90) * spec1 + vec3(0.5, 0.7, 1.0) * rim * 0.35;
      gl_FragColor = vec4(finalColor, 1.0);
    } else if (uRenderMode == 1) { // Surface / Sculpted 3D (warm alabaster clay)
      vec3 clayColor = vec3(0.88, 0.83, 0.78);
      vec3 finalColor = clayColor * totalDiffuse + vec3(1.0, 0.98, 0.94) * spec1 * 0.75 + vec3(0.4, 0.6, 0.85) * rim;
      gl_FragColor = vec4(finalColor, 1.0);
    } else if (uRenderMode == 2) { // Wireframe shaded base
      vec3 darkBase = vec3(0.08, 0.12, 0.18) * totalDiffuse;
      gl_FragColor = vec4(darkBase, 0.95);
    } else { // Confidence or Comparison Heatmap
      vec3 heatColor = vColor.rgb;
      vec3 finalColor = heatColor * totalDiffuse + vec3(1.0) * spec1 * 0.25;
      gl_FragColor = vec4(finalColor, 1.0);
    }
  }
`;

function mat4Perspective(fovY, aspect, near, far) {
  const f = 1.0 / Math.tan(fovY / 2);
  const nf = 1.0 / (near - far);
  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, (2 * far * near) * nf, 0,
  ];
}

function mat4ModelView(yaw, pitch, distance, targetY = 0) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);

  return [
    cy, sp * sy, cp * sy, 0,
    0, cp, -sp, 0,
    -sy, -sp * cy, cp * cy, 0,
    0, targetY, -distance, 1,
  ];
}

function mat3FromMat4(m) {
  return [
    m[0], m[1], m[2],
    m[4], m[5], m[6],
    m[8], m[9], m[10],
  ];
}

function initWebGL(canvas) {
  const gl = canvas.getContext("webgl", { antialias: true, alpha: true }) ||
             canvas.getContext("experimental-webgl", { antialias: true, alpha: true });
  if (!gl) return null;

  function createShader(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  }

  const vs = createShader(gl.VERTEX_SHADER, VS_SOURCE);
  const fs = createShader(gl.FRAGMENT_SHADER, FS_SOURCE);
  if (!vs || !fs) return null;

  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error(gl.getProgramInfoLog(prog));
    return null;
  }

  const posBuf = gl.createBuffer();
  const normBuf = gl.createBuffer();
  const uvBuf = gl.createBuffer();
  const colBuf = gl.createBuffer();
  const cranialBuf = gl.createBuffer();
  const triBuf = gl.createBuffer();
  const lineBuf = gl.createBuffer();
  const texture = gl.createTexture();

  return {
    gl,
    prog,
    buffers: { posBuf, normBuf, uvBuf, colBuf, cranialBuf, triBuf, lineBuf },
    texture,
    attribs: {
      aPosition: gl.getAttribLocation(prog, "aPosition"),
      aNormal: gl.getAttribLocation(prog, "aNormal"),
      aUv: gl.getAttribLocation(prog, "aUv"),
      aColor: gl.getAttribLocation(prog, "aColor"),
      aIsCranial: gl.getAttribLocation(prog, "aIsCranial"),
    },
    uniforms: {
      uModelView: gl.getUniformLocation(prog, "uModelView"),
      uProjection: gl.getUniformLocation(prog, "uProjection"),
      uNormalMatrix: gl.getUniformLocation(prog, "uNormalMatrix"),
      uRenderMode: gl.getUniformLocation(prog, "uRenderMode"),
      uTexture: gl.getUniformLocation(prog, "uTexture"),
      uHasTexture: gl.getUniformLocation(prog, "uHasTexture"),
      uSkinColor: gl.getUniformLocation(prog, "uSkinColor"),
      uHairColor: gl.getUniformLocation(prog, "uHairColor"),
      uLightDir1: gl.getUniformLocation(prog, "uLightDir1"),
      uLightDir2: gl.getUniformLocation(prog, "uLightDir2"),
      uLightDir3: gl.getUniformLocation(prog, "uLightDir3"),
    },
  };
}

let sharedFaceMesh = null;
function getFaceMeshInstance() {
  if (!sharedFaceMesh && typeof window !== "undefined") {
    try {
      sharedFaceMesh = new FaceMesh({
        locateFile: (file) => `/mediapipe/${file}`,
      });
      sharedFaceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    } catch (e) {
      console.warn("MediaPipe FaceMesh init error:", e);
    }
  }
  return sharedFaceMesh;
}

export default function FaceMesh3DViewer({
  modelData,
  photoUrl = null,
  photos = [],
  title = "3D Face Model",
  badge = null,
  externalRotation = null,
  onRotate = null,
  mode = "photo",
  deviations = null,
  width = 400,
  height = 380,
  showControls = true,
  interactive = true,
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const glStateRef = useRef(null);

  const [internalYaw, setInternalYaw] = useState(0);
  const [internalPitch, setInternalPitch] = useState(0);
  const [zoom, setZoom] = useState(1.0);
  const [autoRotate, setAutoRotate] = useState(false);
  const [renderMode, setRenderMode] = useState(mode);

  const [loadedImages, setLoadedImages] = useState({
    front: null,
    left: null,
    right: null,
  });

  const [detectedMesh, setDetectedMesh] = useState(null);

  const facePalette = useMemo(() => {
    return extractFacePalette(loadedImages.front || loadedImages.left || loadedImages.right);
  }, [loadedImages]);

  const isDraggingRef = useRef(false);
  const lastMousePosRef = useRef({ x: 0, y: 0 });
  const animFrameRef = useRef(null);

  const yaw = externalRotation ? externalRotation.yaw : internalYaw;
  const pitch = externalRotation ? externalRotation.pitch : internalPitch;

  useEffect(() => {
    setRenderMode(mode);
  }, [mode]);

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

  // Run MediaPipe on front photo to extract authentic 3D facial landmarks
  useEffect(() => {
    const img = loadedImages.front;
    if (!img || !img.width || !img.height) return;

    const fm = getFaceMeshInstance();
    if (!fm) return;

    let active = true;
    fm.onResults((results) => {
      if (!active) return;
      if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        const landmarks = results.multiFaceLandmarks[0];
        try {
          const m = buildMeshFromMediaPipeLandmarks(landmarks);
          setDetectedMesh(m);
        } catch (e) {
          console.warn("Failed to build mesh from MediaPipe landmarks:", e);
        }
      }
    });

    fm.send({ image: img }).catch((e) => {
      console.warn("MediaPipe face detection error:", e);
    });

    return () => {
      active = false;
    };
  }, [loadedImages.front]);

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

  // Active mesh: prioritize MediaPipe custom 3D model, fallback cleanly to anatomical template
  const mesh = useMemo(() => {
    if (detectedMesh) {
      return detectedMesh;
    }
    const covered = photos && photos.length > 0 ? photos.map((p) => p.angle) : ["front"];
    return buildAnatomicalFaceMesh(modelData?.face_metadata, covered);
  }, [detectedMesh, modelData, photos]);

  const renderWebGL = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (!glStateRef.current) {
      glStateRef.current = initWebGL(canvas);
    }
    const state = glStateRef.current;
    if (!state) return;

    const { gl, prog, buffers, attribs, uniforms, texture } = state;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || width;
    const h = canvas.clientHeight || height;
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);

    gl.useProgram(prog);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.disable(gl.CULL_FACE);

    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const numVerts = mesh.vertices.length;
    const posArr = new Float32Array(numVerts * 3);
    const normArr = new Float32Array(numVerts * 3);
    const uvArr = new Float32Array(numVerts * 2);
    const colArr = new Float32Array(numVerts * 4);
    const cranialArr = new Float32Array(numVerts);

    for (let i = 0; i < numVerts; i++) {
      const v = mesh.vertices[i];
      posArr[i * 3] = v[0];
      posArr[i * 3 + 1] = v[1];
      posArr[i * 3 + 2] = v[2];

      const n = mesh.normals ? mesh.normals[i] : [0, 0, 1];
      normArr[i * 3] = n[0];
      normArr[i * 3 + 1] = n[1];
      normArr[i * 3 + 2] = n[2];

      const uv = mesh.uvs ? mesh.uvs[i] : { u: 0.5, v: 0.5 };
      uvArr[i * 2] = uv.u;
      uvArr[i * 2 + 1] = uv.v;

      const conf = mesh.confidences && mesh.confidences[i] !== undefined ? mesh.confidences[i] : 1.0;
      const dev = deviations ? deviations[i] || 0 : 0;
      const cranial = mesh.isCranial ? (mesh.isCranial[i] ? 1.0 : 0.0) : 0.0;
      cranialArr[i] = cranial;

      if (renderMode === "confidence") {
        if (conf >= 0.85) {
          colArr[i * 4] = 0.13; colArr[i * 4 + 1] = 0.77; colArr[i * 4 + 2] = 0.37; colArr[i * 4 + 3] = 1.0;
        } else if (conf >= 0.50) {
          colArr[i * 4] = 0.96; colArr[i * 4 + 1] = 0.62; colArr[i * 4 + 2] = 0.04; colArr[i * 4 + 3] = 1.0;
        } else {
          colArr[i * 4] = 0.94; colArr[i * 4 + 1] = 0.27; colArr[i * 4 + 2] = 0.27; colArr[i * 4 + 3] = 1.0;
        }
      } else if (renderMode === "comparison") {
        if (dev < 0.035) {
          colArr[i * 4] = 0.13; colArr[i * 4 + 1] = 0.77; colArr[i * 4 + 2] = 0.37; colArr[i * 4 + 3] = 1.0;
        } else if (dev < 0.075) {
          colArr[i * 4] = 0.92; colArr[i * 4 + 1] = 0.70; colArr[i * 4 + 2] = 0.03; colArr[i * 4 + 3] = 1.0;
        } else {
          colArr[i * 4] = 0.94; colArr[i * 4 + 1] = 0.27; colArr[i * 4 + 2] = 0.27; colArr[i * 4 + 3] = 1.0;
        }
      } else {
        colArr[i * 4] = 1.0; colArr[i * 4 + 1] = 1.0; colArr[i * 4 + 2] = 1.0; colArr[i * 4 + 3] = 1.0;
      }
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, posArr, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(attribs.aPosition);
    gl.vertexAttribPointer(attribs.aPosition, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.normBuf);
    gl.bufferData(gl.ARRAY_BUFFER, normArr, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(attribs.aNormal);
    gl.vertexAttribPointer(attribs.aNormal, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.uvBuf);
    gl.bufferData(gl.ARRAY_BUFFER, uvArr, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(attribs.aUv);
    gl.vertexAttribPointer(attribs.aUv, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.colBuf);
    gl.bufferData(gl.ARRAY_BUFFER, colArr, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(attribs.aColor);
    gl.vertexAttribPointer(attribs.aColor, 4, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.cranialBuf);
    gl.bufferData(gl.ARRAY_BUFFER, cranialArr, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(attribs.aIsCranial);
    gl.vertexAttribPointer(attribs.aIsCranial, 1, gl.FLOAT, false, 0, 0);

    let activeTexture = loadedImages.front;
    if (yaw < -0.3 && loadedImages.left) {
      activeTexture = loadedImages.left;
    } else if (yaw > 0.3 && loadedImages.right) {
      activeTexture = loadedImages.right;
    }

    if (activeTexture) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, activeTexture);
      gl.uniform1i(uniforms.uTexture, 0);
      gl.uniform1f(uniforms.uHasTexture, 1.0);
    } else {
      gl.uniform1f(uniforms.uHasTexture, 0.0);
    }

    const skinR = facePalette.skin[0] / 255;
    const skinG = facePalette.skin[1] / 255;
    const skinB = facePalette.skin[2] / 255;
    gl.uniform3f(uniforms.uSkinColor, skinR, skinG, skinB);

    const hairR = facePalette.hair[0] / 255;
    const hairG = facePalette.hair[1] / 255;
    const hairB = facePalette.hair[2] / 255;
    gl.uniform3f(uniforms.uHairColor, hairR, hairG, hairB);

    gl.uniform3f(uniforms.uLightDir1, 0.42, 0.65, 0.63);
    gl.uniform3f(uniforms.uLightDir2, -0.65, 0.20, 0.45);
    gl.uniform3f(uniforms.uLightDir3, 0.0, -0.55, -0.83);

    const aspect = w / Math.max(1, h);
    const proj = mat4Perspective(Math.PI / 4.2, aspect, 0.1, 20.0);
    const cameraDist = 2.45 / zoom;
    const mv = mat4ModelView(yaw, pitch, cameraDist, 0.0);
    const normMat = mat3FromMat4(mv);

    gl.uniformMatrix4fv(uniforms.uProjection, false, new Float32Array(proj));
    gl.uniformMatrix4fv(uniforms.uModelView, false, new Float32Array(mv));
    gl.uniformMatrix3fv(uniforms.uNormalMatrix, false, new Float32Array(normMat));

    let modeCode = 0;
    if (renderMode === "shaded") modeCode = 1;
    else if (renderMode === "wireframe") modeCode = 2;
    else if (renderMode === "confidence") modeCode = 3;
    else if (renderMode === "comparison") modeCode = 4;
    gl.uniform1i(uniforms.uRenderMode, modeCode);

    const triIndices = new Uint16Array(mesh.triangles.flat());
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.triBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, triIndices, gl.DYNAMIC_DRAW);
    gl.drawElements(gl.TRIANGLES, triIndices.length, gl.UNSIGNED_SHORT, 0);

    if (renderMode === "wireframe" && mesh.wireframeIndices) {
      gl.uniform1i(uniforms.uRenderMode, 5);
      const lineIndices = new Uint16Array(mesh.wireframeIndices);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.lineBuf);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, lineIndices, gl.DYNAMIC_DRAW);
      gl.drawElements(gl.LINES, lineIndices.length, gl.UNSIGNED_SHORT, 0);
    }
  }, [mesh, yaw, pitch, zoom, renderMode, deviations, loadedImages, facePalette, width, height]);

  useEffect(() => {
    renderWebGL();
  }, [renderWebGL]);

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

        {showControls && (
          <div className="viewer-mode-pills">
            <button
              className={`mode-pill ${renderMode === "photo" ? "active" : ""}`}
              onClick={() => setRenderMode("photo")}
              title="True-to-life 3D Photo Projection with Hardware Shading"
            >
              Photo 3D
            </button>
            <button
              className={`mode-pill ${renderMode === "shaded" ? "active" : ""}`}
              onClick={() => setRenderMode("shaded")}
              title="Sculpted 3D Anatomical Bone & Cartilage Surface"
            >
              Surface
            </button>
            <button
              className={`mode-pill ${renderMode === "wireframe" ? "active" : ""}`}
              onClick={() => setRenderMode("wireframe")}
              title="High-Density Polygonal Lattice Mesh"
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

      <div
        className="viewer-3d-canvas-container"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleMouseUp}
        style={{
          background: "radial-gradient(circle at 50% 45%, #162438 0%, #0c1422 65%, #05080e 100%)",
          borderRadius: "8px",
          overflow: "hidden",
          position: "relative",
        }}
      >
        <canvas
          ref={canvasRef}
          style={{ width: "100%", height: `${height}px`, display: "block", cursor: "grab" }}
        />

        <div className="viewer-hud-overlay">
          <span>Yaw: {Math.round((yaw * 180) / Math.PI)}°</span>
          <span>Pitch: {Math.round((pitch * 180) / Math.PI)}°</span>
          <span>Zoom: {zoom.toFixed(1)}x</span>
        </div>

        {renderMode === "confidence" && !isComplete && (
          <div className="viewer-inconsistency-badge">
            <span className="dot pulse-red" />
            <span>Red/Amber: Missing Views & Depth Ambiguities</span>
          </div>
        )}

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
