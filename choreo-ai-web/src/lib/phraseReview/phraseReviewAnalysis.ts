/**
 * Analyze recorded user keyframes vs reference skeleton timeline for phrase review moments.
 */

import type { SkeletonFrame } from "@/lib/phraseReview/types";
import {
  JOINT_ANGLES,
  calculateJointAngles,
  landmarkMapFromList,
} from "@/lib/pose/skeletonPrimitives";

export type PhraseMomentSnapshot = {
  timestamp_ms: number;
  worstJointKey: string;
  worstJointLabel: string;
  totalDiffDeg: number;
  userDeg: number;
  refDeg: number;
  userJoints: Record<string, { x: number; y: number }>;
  refJoints: Record<string, { x: number; y: number }>;
};

/** User webcam snapshot keyed to reference timeline (video `currentTime` ms). */
export type UserPhraseKeyframe = {
  timestamp_ms: number;
  joints: Record<string, { x: number; y: number }>;
};

/** Closest pose keyframe by reference-video timestamp — same heuristic as chunk attempt replay sync. */
export function nearestUserKeyframe(
  keyframes: UserPhraseKeyframe[],
  targetCaptureMs: number
): UserPhraseKeyframe | null {
  if (!keyframes.length) return null;
  let best = keyframes[0]!;
  let bestDiff = Math.abs(best.timestamp_ms - targetCaptureMs);
  for (let i = 1; i < keyframes.length; i++) {
    const kf = keyframes[i]!;
    const diff = Math.abs(kf.timestamp_ms - targetCaptureMs);
    if (diff < bestDiff) {
      best = kf;
      bestDiff = diff;
    }
  }
  return best && Object.keys(best.joints).length > 0 ? best : null;
}

const LR_PAIR_NAMES: [string, string][] = [
  ["LEFT_SHOULDER", "RIGHT_SHOULDER"],
  ["LEFT_ELBOW", "RIGHT_ELBOW"],
  ["LEFT_WRIST", "RIGHT_WRIST"],
  ["LEFT_HIP", "RIGHT_HIP"],
  ["LEFT_KNEE", "RIGHT_KNEE"],
  ["LEFT_ANKLE", "RIGHT_ANKLE"],
];

export function swapLeftRightJointAngles(angles: Record<string, number>): Record<string, number> {
  const swapped: Record<string, number> = {};
  for (const { name } of JOINT_ANGLES) {
    if (name.startsWith("LEFT_")) {
      const rhs = `RIGHT_${name.slice(5)}`;
      swapped[name] = angles[rhs] ?? angles[name];
    } else if (name.startsWith("RIGHT_")) {
      const lhs = `LEFT_${name.slice(6)}`;
      swapped[name] = angles[lhs] ?? angles[name];
    }
  }
  return swapped;
}

/** Swap XY landmark positions left/right */
export function swapLeftRightPositions(joints: Record<string, { x: number; y: number }>): Record<
  string,
  { x: number; y: number }
> {
  const out: Record<string, { x: number; y: number }> = { ...joints };
  for (const [a, b] of LR_PAIR_NAMES) {
    const pa = joints[a];
    const pb = joints[b];
    if (pa && pb) {
      out[a] = pb;
      out[b] = pa;
    }
  }
  return out;
}

export function interpolatedAnglesAtTime(
  frames: SkeletonFrame[],
  timeMs: number
): Record<string, number> {
  if (!frames.length) return {};
  let left: SkeletonFrame | null = null;
  let right: SkeletonFrame | null = null;
  for (const f of frames) {
    const t = f.timestamp_ms ?? 0;
    if (t <= timeMs) left = f;
    if (t >= timeMs) {
      right = f;
      break;
    }
  }
  const l = left ?? frames[0];
  const r = right ?? frames[frames.length - 1];
  const lt = l.timestamp_ms ?? timeMs;
  const rt = r.timestamp_ms ?? timeMs;
  const la = calculateJointAngles(landmarkMapFromList(l.landmarks ?? []));
  const ra = calculateJointAngles(landmarkMapFromList(r.landmarks ?? []));
  if (lt === rt) return la;
  const alpha = Math.max(0, Math.min(1, (timeMs - lt) / (rt - lt)));
  const out: Record<string, number> = {};
  for (const { name } of JOINT_ANGLES) {
    const lv = la[name];
    const rv = ra[name];
    if (lv === undefined || rv === undefined) continue;
    out[name] = lv + (rv - lv) * alpha;
  }
  return out;
}

function interpNum(a: number, b: number, alpha: number) {
  return a + (b - a) * alpha;
}

/** Interpolate reference landmark XY at timeline `timeMs` */
export function interpolatedJointsFromSkeleton(
  frames: SkeletonFrame[],
  timeMs: number
): Record<string, { x: number; y: number }> {
  if (!frames.length) return {};
  let left: SkeletonFrame | null = null;
  let right: SkeletonFrame | null = null;
  for (const f of frames) {
    const t = f.timestamp_ms ?? 0;
    if (t <= timeMs) left = f;
    if (t >= timeMs) {
      right = f;
      break;
    }
  }
  const l = left ?? frames[0];
  const r = right ?? frames[frames.length - 1];
  const lm = landmarkMapFromList(l.landmarks ?? []);
  const rm = landmarkMapFromList(r.landmarks ?? []);
  const lt = l.timestamp_ms ?? timeMs;
  const rt = r.timestamp_ms ?? timeMs;
  if (lt === rt) return { ...lm };

  const alpha = Math.max(0, Math.min(1, (timeMs - lt) / (rt - lt)));
  const keys = new Set([...Object.keys(lm), ...Object.keys(rm)]);
  const out: Record<string, { x: number; y: number }> = {};
  for (const k of keys) {
    const a = lm[k];
    const b = rm[k];
    if (a && b) out[k] = { x: interpNum(a.x, b.x, alpha), y: interpNum(a.y, b.y, alpha) };
    else if (a) out[k] = { ...a };
    else if (b) out[k] = { ...b };
  }
  return out;
}

/**
 * Frames in buffer must use reference video timestamps.
 * Finds top frames by sum of joint angle deltas vs interpolated reference at that time.
 */
export function computeWorstMoments(params: {
  userKeyframes: UserPhraseKeyframe[];
  skeletonFrames: SkeletonFrame[];
  phraseStartMs: number;
  phraseEndMs: number;
  mirrorMapping: "direct" | "lr_swapped";
  topN?: number;
}): PhraseMomentSnapshot[] {
  const { userKeyframes, skeletonFrames, phraseStartMs, phraseEndMs, mirrorMapping } = params;
  const topN = params.topN ?? 3;

  const inPhrase = userKeyframes.filter(
    (k) =>
      k.timestamp_ms >= phraseStartMs - 12 &&
      k.timestamp_ms <= phraseEndMs + 12 &&
      Object.keys(k.joints).length > 4
  );
  if (!skeletonFrames.length || inPhrase.length === 0) return [];

  const scored: PhraseMomentSnapshot[] = [];

  for (const kf of inPhrase) {
    const t = Math.min(Math.max(kf.timestamp_ms, phraseStartMs), phraseEndMs);
    let refAngles = interpolatedAnglesAtTime(skeletonFrames, t);
    if (mirrorMapping === "lr_swapped") refAngles = swapLeftRightJointAngles(refAngles);

    const refJointsRaw = interpolatedJointsFromSkeleton(skeletonFrames, t);
    const refJointsDraw =
      mirrorMapping === "lr_swapped"
        ? swapLeftRightPositions(refJointsRaw)
        : refJointsRaw;

    const userAngles = calculateJointAngles(kf.joints);

    let totalDiff = 0;
    let worstJointKey = JOINT_ANGLES[0]!.name;
    let worstJointLabel = JOINT_ANGLES[0]!.label;
    let worstJointDiff = -1;

    for (const { name, label } of JOINT_ANGLES) {
      const u = userAngles[name];
      const rr = refAngles[name];
      if (u === undefined || rr === undefined) continue;
      const d = Math.abs(u - rr);
      totalDiff += d;
      if (d > worstJointDiff) {
        worstJointDiff = d;
        worstJointKey = name;
        worstJointLabel = label;
      }
    }

    const ud = worstJointDiff >= 0 ? userAngles[worstJointKey] : undefined;
    const rd = worstJointDiff >= 0 ? refAngles[worstJointKey] : undefined;

    scored.push({
      timestamp_ms: t,
      worstJointKey,
      worstJointLabel,
      totalDiffDeg: Math.round(totalDiff * 100) / 100,
      userDeg: ud != null ? Math.round(ud * 10) / 10 : 0,
      refDeg: rd != null ? Math.round(rd * 10) / 10 : 0,
      userJoints: { ...kf.joints },
      refJoints: { ...refJointsDraw },
    });
  }

  scored.sort((a, b) => b.totalDiffDeg - a.totalDiffDeg || b.timestamp_ms - a.timestamp_ms);

  const uniq: PhraseMomentSnapshot[] = [];
  for (const row of scored) {
    if (uniq.some((u) => Math.abs(u.timestamp_ms - row.timestamp_ms) < 280)) continue;
    uniq.push(row);
    if (uniq.length >= topN) break;
  }
  return uniq;
}
