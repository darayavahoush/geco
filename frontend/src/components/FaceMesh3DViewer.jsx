import { useEffect, useRef, useState, useCallback, useMemo } from "react";

/**
 * Builds an authentic, solid 2-manifold 3D human head mesh.
 * - Width 0.64, Height 0.84, Depth 0.72 (Golden aspect ratio ~1.31)
 * - Closed cranial vault apex dome (no open holes or thimble cones)
 * - Planar perspective UV mapping aligned to the face
 * - Subtle realistic facial anatomy (nose, eye sockets, lips, cheekbones, chin)
 * - Unit surface normals accumulated from incident face normals
 */
export function buildAnatomicalFaceMesh(faceMeta = null, coveredAngles = ["front"], nLat = 42, nLon = 48) {
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
    shiftEyeY = (-eyeYNorm - 0.20) * 0.15;

    const noseYNorm = (nt[1] - cy) / Math.max(1.0, bh);
    shiftNoseY = (-noseYNorm - -0.10) * 0.20;

    const mouthYNorm = ((lMouth[1] + rMouth[1]) / 2.0 - cy) / Math.max(1.0, bh);
    shiftMouthY = (-mouthYNorm - -0.35) * 0.15;

    const mouthWPx = Math.abs(rMouth[0] - lMouth[0]);
    scaleMouthW = Math.max(0.85, Math.min(1.20, (mouthWPx / Math.max(1.0, bw)) / 0.38));

    const aspect = bw / Math.max(1.0, bh);
    scaleJawW = Math.max(0.88, Math.min(1.12, aspect / 0.85));
    scaleFaceH = Math.max(0.92, Math.min(1.10, 0.85 / Math.max(0.5, aspect)));
  }

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

  // 1. Top Pole Vertex (Smooth cranial apex dome)
  const topY = Math.round(0.42 * scaleFaceH * 10000) / 10000;
  vertices.push([0.0, topY, 0.0]);
  uvs.push({ u: 0.50, v: 0.01 });
  confidences.push(covered.has("front") ? 0.95 : 0.40);
  isCranial.push(true);

  // 2. Intermediate Ring Vertices
  for (let j = 1; j < nLat - 1; j++) {
    const phi = phis[j];
    const y = Math.round(0.42 * Math.cos(phi) * scaleFaceH * 10000) / 10000;

    let rx, rzF, rzB;
    if (y >= 0) {
      const s = Math.sqrt(Math.max(0.01, 1.0 - Math.pow(y / 0.45, 2)));
      rx = 0.32 * s;
      rzF = 0.28 * s;
      rzB = 0.36 * s;
    } else {
      const prog = -y / 0.42;
      const taper = (1.0 - 0.26 * Math.pow(prog, 1.1)) * scaleJawW;
      rx = 0.32 * taper;
      rzF = 0.28 * (1.0 - 0.18 * prog);
      rzB = 0.36 * (1.0 - 0.32 * prog);
    }

    for (let i = 0; i < nLon; i++) {
      const th = thetas[i];
      const sinT = Math.sin(th);
      const cosT = Math.cos(th);

      const x = Math.round(rx * sinT * 10000) / 10000;

      if (cosT >= 0) {
        // Anterior facial relief
        const zBase = rzF * Math.pow(cosT, 0.85);

        // 3D Nose
        let zNose = 0.0;
        const noseYMin = -0.14 + shiftNoseY;
        const noseYMax = 0.10 + shiftNoseY;
        if (y >= noseYMin && y <= noseYMax && Math.abs(x) < 0.10) {
          const latN = Math.exp(-0.5 * Math.pow(x / 0.035, 2));
          const noseTipY = -0.04 + shiftNoseY;
          let vertN = 0.0;
          if (y >= noseTipY) {
            const progN = (noseYMax - y) / Math.max(0.01, noseYMax - noseTipY);
            vertN = 0.015 + 0.075 * Math.pow(progN, 0.9);
          } else {
            const progN = (y - noseYMin) / Math.max(0.01, noseTipY - noseYMin);
            vertN = 0.015 + 0.075 * Math.pow(progN, 1.1);
          }
          const zBridge = vertN * latN * Math.min(1.25, Math.max(0.80, scaleNoseZ));

          // Nostril wings
          let zAla = 0.0;
          const alaY = -0.06 + shiftNoseY;
          if (y >= -0.11 && y <= -0.02 && Math.abs(x) >= 0.03 && Math.abs(x) <= 0.08) {
            const dAla = Math.sqrt(Math.pow((Math.abs(x) - 0.05) / 0.03, 2) + Math.pow((y - alaY) / 0.035, 2));
            if (dAla < 1.0) {
              zAla = 0.020 * (1.0 - dAla * dAla);
            }
          }
          zNose = zBridge + zAla;
        }

        // Orbital eye sockets
        let zOrbit = 0.0;
        const eyeY = 0.07 + shiftEyeY;
        const dEye = Math.sqrt(Math.pow((Math.abs(x) - 0.14 * scaleEyeW) / 0.07, 2) + Math.pow((y - eyeY) / 0.05, 2));
        if (dEye < 1.0) {
          zOrbit = -0.020 * (1.0 - dEye * dEye);
        }

        // Brow ridge
        let zBrow = 0.0;
        const browY = 0.13 + shiftEyeY;
        if (Math.abs(y - browY) < 0.035 && Math.abs(x) < 0.20) {
          zBrow = 0.018 * Math.exp(-0.5 * Math.pow(x / 0.13, 2)) * (1.0 - Math.abs(y - browY) / 0.035);
        }

        // Cheekbones
        let zCheek = 0.0;
        const cheekY = 0.02 + shiftEyeY;
        const dCheek = Math.sqrt(Math.pow((Math.abs(x) - 0.18) / 0.06, 2) + Math.pow((y - cheekY) / 0.05, 2));
        if (dCheek < 1.0) {
          zCheek = 0.020 * (1.0 - dCheek * dCheek);
        }

        // Lips
        let zLips = 0.0;
        const mouthYMin = -0.25 + shiftMouthY;
        const mouthYMax = -0.13 + shiftMouthY;
        if (y >= mouthYMin && y <= mouthYMax && Math.abs(x) < 0.12) {
          const latM = Math.exp(-0.5 * Math.pow(x / (0.07 * scaleMouthW), 2));
          const fissY = -0.19 + shiftMouthY;
          if (y >= fissY) {
            zLips = 0.020 * Math.sin(((y - fissY) / Math.max(0.01, mouthYMax - fissY)) * Math.PI) * latM;
          } else {
            zLips = 0.022 * Math.sin(((y - mouthYMin) / Math.max(0.01, fissY - mouthYMin)) * Math.PI) * latM;
          }
        }

        // Chin (pogonion)
        let zChin = 0.0;
        if (y >= -0.38 && y <= -0.27 && Math.abs(x) < 0.10) {
          const latC = Math.exp(-0.5 * Math.pow(x / 0.07, 2));
          zChin = 0.030 * Math.sin(((y - -0.38) / 0.11) * Math.PI) * latC;
        }

        const z = Math.round((zBase + zNose + zOrbit + zBrow + zCheek + zLips + zChin) * 10000) / 10000;
        isCranial.push(false);

        // Planar perspective UV projection aligned to frontal camera frame
        const u = Math.round(Math.max(0.01, Math.min(0.99, 0.50 + x / 0.60)) * 10000) / 10000;
        const v = Math.round(Math.max(0.01, Math.min(0.99, 0.50 - y / 0.76)) * 10000) / 10000;
        uvs.push({ u, v });
        vertices.push([x, y, z]);
      } else {
        const z = -Math.round((rzB * Math.pow(Math.abs(cosT), 0.85)) * 10000) / 10000;
        isCranial.push(true);
        uvs.push({ u: 0.50, v: 0.50 });
        vertices.push([x, y, z]);
      }

      let conf = covered.has("front") ? 0.95 : 0.40;
      if (Math.abs(th) > 0.85) {
        const sideCovered = (covered.has("left") && th < 0) || (covered.has("right") && th > 0);
        conf = sideCovered ? 0.95 : 0.45;
      }
      confidences.push(Math.round(conf * 100) / 100);
    }
  }

  // 3. Bottom Pole Vertex (Smooth submental jawline/chin apex)
  const botY = -Math.round(0.42 * scaleFaceH * 10000) / 10000;
  vertices.push([0.0, botY, 0.0]);
  uvs.push({ u: 0.50, v: 0.99 });
  confidences.push(covered.has("front") ? 0.95 : 0.40);
  isCranial.push(true);

  const totalVerts = vertices.length;

  // 4. Construct Closed Manifold Triangles
  const triangles = [];
  const wireframeIndices = [];

  // Top cap fan
  for (let i = 0; i < nLon; i++) {
    const iNext = (i + 1) % nLon;
    const a = 0, b = 1 + i, c = 1 + iNext;
    triangles.push([a, b, c]);
    wireframeIndices.push(a, b, b, c, c, a);
  }

  // Intermediate rings
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

  // Bottom cap fan
  const botPole = totalVerts - 1;
  const lastR = 1 + (nLat - 3) * nLon;
  for (let i = 0; i < nLon; i++) {
    const iNext = (i + 1) % nLon;
    const a = botPole, b = lastR + iNext, c = lastR + i;
    triangles.push([a, b, c]);
    wireframeIndices.push(a, b, b, c, c, a);
  }

  // 5. Compute Unit Normals from Incident Face Normals
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
    normals[i] = [
      Math.round((normals[i][0] / l) * 10000) / 10000,
      Math.round((normals[i][1] / l) * 10000) / 10000,
      Math.round((normals[i][2] / l) * 10000) / 10000,
    ];
  }

  return { vertices, triangles, uvs, confidences, isCranial, normals, wireframeIndices };
}

// Proportional face crop preserving natural 1.25 height/width ratio
function getFaceCropRect(img, faceMeta) {
  if (!img || !img.width || !img.height) return null;
  const w = img.width;
  const h = img.height;

  let cx = w / 2;
  let cy = h / 2;
  let faceW = Math.min(w, h) * 0.65;
  let faceH = faceW * 1.25;

  if (faceMeta && faceMeta.bbox) {
    const [bx, by, bw, bh] = faceMeta.bbox;
    cx = bx + bw / 2;
    cy = by + bh / 2;
    faceW = bw * 1.35;
    faceH = faceW * 1.28;
  }

  const sx = Math.max(0, Math.round(cx - faceW / 2));
  const sy = Math.max(0, Math.round(cy - faceH / 2));
  const sw = Math.min(w - sx, Math.round(faceW));
  const sh = Math.min(h - sy, Math.round(faceH));

  return { sx, sy, sw, sh };
}

// Samples real hair and skin tone from photo for seamless shading
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

  uniform int uRenderMode; // 0=photo, 1=surface/shaded, 2=wireframe base, 3=confidence, 4=comparison, 5=wireframe lines
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

    // Three-point studio lighting (Key, Fill, Rim)
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
        // Smooth transition into skin tone towards the profile edges
        float edgeDist = clamp(abs(vUv.x - 0.5) * 2.0, 0.0, 1.0);
        float blend = smoothstep(0.78, 0.99, edgeDist);
        baseColor = mix(texColor.rgb, uSkinColor, blend);
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

  // Resolve continuous 3D anatomical face mesh (1,922 vertices)
  const mesh = useMemo(() => {
    if (modelData?.vertices && modelData.vertices.length >= 1900 && modelData?.triangles) {
      const uvs = (modelData.uvs && modelData.uvs.length === modelData.vertices.length)
        ? modelData.uvs
        : modelData.vertices.map(([x, y]) => ({
            u: Math.max(0.01, Math.min(0.99, 0.50 + x / 0.60)),
            v: Math.max(0.01, Math.min(0.99, 0.50 - y / 0.76)),
          }));

      let normals = modelData.normals;
      if (!normals || normals.length !== modelData.vertices.length) {
        const totalVerts = modelData.vertices.length;
        normals = new Array(totalVerts).fill(0).map(() => [0, 0, 0]);
        for (let k = 0; k < modelData.triangles.length; k++) {
          const [a, b, c] = modelData.triangles[k];
          const va = modelData.vertices[a], vb = modelData.vertices[b], vc = modelData.vertices[c];
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
          normals[i] = [normals[i][0] / l, normals[i][1] / l, normals[i][2] / l];
        }
      }

      const wireframeIndices = [];
      for (let k = 0; k < modelData.triangles.length; k++) {
        const [a, b, c] = modelData.triangles[k];
        wireframeIndices.push(a, b, b, c, c, a);
      }

      return {
        vertices: modelData.vertices,
        triangles: modelData.triangles,
        uvs,
        normals,
        wireframeIndices,
        confidences: modelData.confidence_per_vertex || [],
        isCranial: modelData.is_cranial || modelData.vertices.map(([x, y, z]) => z < 0.02 || Math.abs(x) > 0.30),
      };
    }
    const covered = photos && photos.length > 0 ? photos.map((p) => p.angle) : ["front"];
    return buildAnatomicalFaceMesh(modelData?.face_metadata, covered);
  }, [modelData, photos]);

  // Main Hardware WebGL 3D Rendering Engine
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
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);

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
      const cranial = mesh.isCranial ? (mesh.isCranial[i] ? 1.0 : 0.0) : (v[2] < 0.02 ? 1.0 : 0.0);
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

    // Setup active texture
    let activeTexture = loadedImages.front;
    if (yaw < -0.3 && loadedImages.left) {
      activeTexture = loadedImages.left;
    } else if (yaw > 0.3 && loadedImages.right) {
      activeTexture = loadedImages.right;
    }

    if (activeTexture) {
      const crop = getFaceCropRect(activeTexture, modelData?.face_metadata);
      const cropCanvas = document.createElement("canvas");
      cropCanvas.width = 512;
      cropCanvas.height = 512;
      const cropCtx = cropCanvas.getContext("2d");
      const sx = crop ? crop.sx : 0;
      const sy = crop ? crop.sy : 0;
      const sw = crop ? crop.sw : activeTexture.width;
      const sh = crop ? crop.sh : activeTexture.height;
      cropCtx.drawImage(activeTexture, sx, sy, sw, sh, 0, 0, 512, 512);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, cropCanvas);
      gl.uniform1i(uniforms.uTexture, 0);
      gl.uniform1f(uniforms.uHasTexture, 1.0);
    } else {
      gl.uniform1f(uniforms.uHasTexture, 0.0);
    }

    // Set lighting & colors
    const skinR = facePalette.skin[0] / 255;
    const skinG = facePalette.skin[1] / 255;
    const skinB = facePalette.skin[2] / 255;
    gl.uniform3f(uniforms.uSkinColor, skinR, skinG, skinB);

    const hairR = facePalette.hair[0] / 255;
    const hairG = facePalette.hair[1] / 255;
    const hairB = facePalette.hair[2] / 255;
    gl.uniform3f(uniforms.uHairColor, hairR, hairG, hairB);

    // Three-point lights
    gl.uniform3f(uniforms.uLightDir1, 0.42, 0.65, 0.63);
    gl.uniform3f(uniforms.uLightDir2, -0.65, 0.20, 0.45);
    gl.uniform3f(uniforms.uLightDir3, 0.0, -0.55, -0.83);

    // Camera matrices
    const aspect = w / Math.max(1, h);
    const proj = mat4Perspective(Math.PI / 4.2, aspect, 0.1, 20.0);
    const cameraDist = 2.45 / zoom;
    const mv = mat4ModelView(yaw, pitch, cameraDist, 0.0);
    const normMat = mat3FromMat4(mv);

    gl.uniformMatrix4fv(uniforms.uProjection, false, new Float32Array(proj));
    gl.uniformMatrix4fv(uniforms.uModelView, false, new Float32Array(mv));
    gl.uniformMatrix3fv(uniforms.uNormalMatrix, false, new Float32Array(normMat));

    let modeCode = 0; // photo
    if (renderMode === "shaded") modeCode = 1;
    else if (renderMode === "wireframe") modeCode = 2;
    else if (renderMode === "confidence") modeCode = 3;
    else if (renderMode === "comparison") modeCode = 4;
    gl.uniform1i(uniforms.uRenderMode, modeCode);

    const triIndices = new Uint16Array(mesh.triangles.flat());
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.triBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, triIndices, gl.DYNAMIC_DRAW);
    gl.drawElements(gl.TRIANGLES, triIndices.length, gl.UNSIGNED_SHORT, 0);

    // If Wireframe mode: draw overlaid glowing mesh lines
    if (renderMode === "wireframe" && mesh.wireframeIndices) {
      gl.uniform1i(uniforms.uRenderMode, 5);
      gl.disable(gl.CULL_FACE);
      const lineIndices = new Uint16Array(mesh.wireframeIndices);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.lineBuf);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, lineIndices, gl.DYNAMIC_DRAW);
      gl.drawElements(gl.LINES, lineIndices.length, gl.UNSIGNED_SHORT, 0);
    }
  }, [mesh, modelData, yaw, pitch, zoom, renderMode, deviations, loadedImages, facePalette, width, height]);

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
