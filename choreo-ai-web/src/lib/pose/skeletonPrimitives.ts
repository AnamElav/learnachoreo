/**
 * Shared skeleton drawing primitives for canvases (player + PhraseReview).
 */

/** Key joints for stick figure */
export const POSE_JOINTS = new Set([
  "NOSE",
  "LEFT_SHOULDER",
  "RIGHT_SHOULDER",
  "LEFT_ELBOW",
  "RIGHT_ELBOW",
  "LEFT_WRIST",
  "RIGHT_WRIST",
  "LEFT_HIP",
  "RIGHT_HIP",
  "LEFT_KNEE",
  "RIGHT_KNEE",
  "LEFT_ANKLE",
  "RIGHT_ANKLE",
]);

export const POSE_EDGES: [string, string][] = [
  ["NOSE", "LEFT_SHOULDER"],
  ["NOSE", "RIGHT_SHOULDER"],
  ["LEFT_SHOULDER", "RIGHT_SHOULDER"],
  ["LEFT_SHOULDER", "LEFT_ELBOW"],
  ["LEFT_ELBOW", "LEFT_WRIST"],
  ["RIGHT_SHOULDER", "RIGHT_ELBOW"],
  ["RIGHT_ELBOW", "RIGHT_WRIST"],
  ["LEFT_SHOULDER", "LEFT_HIP"],
  ["RIGHT_SHOULDER", "RIGHT_HIP"],
  ["LEFT_HIP", "RIGHT_HIP"],
  ["LEFT_HIP", "LEFT_KNEE"],
  ["LEFT_KNEE", "LEFT_ANKLE"],
  ["RIGHT_HIP", "RIGHT_KNEE"],
  ["RIGHT_KNEE", "RIGHT_ANKLE"],
];

/** Joint angles (vertex middle point) — same subset as player page */
export const JOINT_ANGLES: { name: string; label: string; points: [string, string, string] }[] = [
  { name: "LEFT_ELBOW", label: "left elbow", points: ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"] },
  { name: "RIGHT_ELBOW", label: "right elbow", points: ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"] },
  { name: "LEFT_KNEE", label: "left knee", points: ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"] },
  { name: "RIGHT_KNEE", label: "right knee", points: ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"] },
  { name: "LEFT_SHOULDER", label: "left shoulder", points: ["LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"] },
  { name: "RIGHT_SHOULDER", label: "right shoulder", points: ["RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"] },
];

export function calculateAngleDeg(
  a: { x: number; y: number },
  b: { x: number; y: number },
  c: { x: number; y: number }
): number {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const magAB = Math.sqrt(ab.x * ab.x + ab.y * ab.y);
  const magCB = Math.sqrt(cb.x * cb.x + cb.y * cb.y);
  if (magAB === 0 || magCB === 0) return 0;
  const cosAngle = Math.max(-1, Math.min(1, dot / (magAB * magCB)));
  return Math.acos(cosAngle) * (180 / Math.PI);
}

export function landmarkMapFromList(landmarks: { name: string; x: number; y: number }[]): Record<
  string,
  { x: number; y: number }
> {
  const m: Record<string, { x: number; y: number }> = {};
  for (const lm of landmarks) {
    m[lm.name] = { x: lm.x, y: lm.y };
  }
  return m;
}

export function calculateJointAngles(joints: Record<string, { x: number; y: number }>): Record<string, number> {
  const angles: Record<string, number> = {};
  for (const { name, points } of JOINT_ANGLES) {
    const [aName, bName, cName] = points;
    const a = joints[aName];
    const b = joints[bName];
    const c = joints[cName];
    if (a && b && c) angles[name] = calculateAngleDeg(a, b, c);
  }
  return angles;
}

/** Draw skeleton; `toCanvas` maps normalized skeleton coords → canvas px */
export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  joints: Record<string, { x: number; y: number }>,
  toCanvas: (p: { x: number; y: number }) => { x: number; y: number },
  strokeColor: string,
  fillColor: string,
  opts?: {
    focusJoint?: string | null;
    highlightColor?: string;
    baseLineWidth?: number;
    jointRadius?: number;
  }
) {
  const highlight = opts?.focusJoint ?? null;
  const hi = opts?.highlightColor ?? "#FBBF24";
  const baseLw = opts?.baseLineWidth ?? 2;
  const jointRadius = opts?.jointRadius ?? 4;

  ctx.lineCap = "round";
  for (const [a, b] of POSE_EDGES) {
    const pa = joints[a];
    const pb = joints[b];
    if (!pa || !pb) continue;
    const ca = toCanvas(pa);
    const cb = toCanvas(pb);
    const touches = highlight && (a === highlight || b === highlight);
    ctx.strokeStyle = touches ? hi : strokeColor;
    ctx.lineWidth = touches ? baseLw + 2 : baseLw;
    ctx.beginPath();
    ctx.moveTo(ca.x, ca.y);
    ctx.lineTo(cb.x, cb.y);
    ctx.stroke();
  }

  for (const name of POSE_JOINTS) {
    const p = joints[name];
    if (!p) continue;
    const c = toCanvas(p);
    const isFocus = highlight === name;
    ctx.fillStyle = isFocus ? hi : fillColor;
    ctx.beginPath();
    ctx.arc(c.x, c.y, isFocus ? jointRadius + 3 : jointRadius, 0, Math.PI * 2);
    ctx.fill();
  }
}

/** Fit normalized joints into w×h with padding */
export function makeFitNormalizer(
  joints: Record<string, { x: number; y: number }>,
  width: number,
  height: number,
  pad = 14
): (p: { x: number; y: number }) => { x: number; y: number } {
  let minX = 1,
    maxX = 0,
    minY = 1,
    maxY = 0;
  let any = false;
  for (const name of POSE_JOINTS) {
    const p = joints[name];
    if (!p) continue;
    any = true;
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  if (!any) return () => ({ x: width / 2, y: height / 2 });

  const bw = Math.max(1e-6, maxX - minX);
  const bh = Math.max(1e-6, maxY - minY);
  const cw = width - pad * 2;
  const ch = height - pad * 2;
  const scale = Math.min(cw / bw, ch / bh);
  const drawW = bw * scale;
  const drawH = bh * scale;
  const ox = (width - drawW) / 2 - minX * scale;
  const oy = (height - drawH) / 2 - minY * scale;

  return (p: { x: number; y: number }) => ({
    x: p.x * scale + ox,
    y: p.y * scale + oy,
  });
}
