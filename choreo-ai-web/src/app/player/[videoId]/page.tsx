"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useSearchParams, useRouter } from "next/navigation";
import { PhraseReview } from "@/components/PhraseReview";
import { supabase } from "@/lib/supabase/client";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Landmark {
  name: string;
  x: number;
  y: number;
  z?: number;
  visibility?: number;
}

interface SkeletonFrame {
  frame_number?: number;
  timestamp_ms?: number;
  landmarks?: Landmark[];
}

interface Segment {
  segment_id: number;
  start_frame?: number;
  end_frame?: number;
  start_time_ms: number;
  end_time_ms: number;
  beat_count: number;
  angle_summary?: Record<string, number>;
}

interface ChoreoData {
  video_id?: string;
  quality_assessment?: { overall_score?: number };
  segments?: Segment[];
}

type RecordedAttemptKeyframe = {
  timestamp_ms: number;
  joints: Record<string, { x: number; y: number }>;
};

function smoothJointsEma(
  current: Record<string, { x: number; y: number }>,
  prev: Record<string, { x: number; y: number }> | null,
  alpha = 0.3
): Record<string, { x: number; y: number }> {
  if (!prev) return { ...current };
  const out: Record<string, { x: number; y: number }> = {};
  for (const [name, p] of Object.entries(current)) {
    const pp = prev[name];
    if (!pp) {
      out[name] = { ...p };
      continue;
    }
    out[name] = {
      x: alpha * p.x + (1 - alpha) * pp.x,
      y: alpha * p.y + (1 - alpha) * pp.y,
    };
  }
  return out;
}

/** Key joints for stick figure (excludes face/ear landmarks to avoid overlapping head circles). */
const POSE_JOINTS = new Set([
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

/** MediaPipe pose edges for stick figure (landmark names). */
const POSE_EDGES: [string, string][] = [
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

/** MoveNet keypoint indices to names (17 keypoints) */
const MOVENET_KEYPOINTS = [
  "NOSE",
  "LEFT_EYE",
  "RIGHT_EYE",
  "LEFT_EAR",
  "RIGHT_EAR",
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
];

function landmarkMap(landmarks: Landmark[]): Record<string, { x: number; y: number }> {
  const m: Record<string, { x: number; y: number }> = {};
  for (const lm of landmarks) {
    m[lm.name] = { x: lm.x, y: lm.y };
  }
  return m;
}

/** Calculate angle at point B given three points A, B, C (in degrees) */
function calculateAngle(
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

/** Joint angle definitions: [jointName, pointA, pointB (vertex), pointC] */
const JOINT_ANGLES: { name: string; label: string; points: [string, string, string] }[] = [
  { name: "LEFT_ELBOW", label: "left elbow", points: ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"] },
  { name: "RIGHT_ELBOW", label: "right elbow", points: ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"] },
  { name: "LEFT_KNEE", label: "left knee", points: ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"] },
  { name: "RIGHT_KNEE", label: "right knee", points: ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"] },
  { name: "LEFT_SHOULDER", label: "left shoulder", points: ["LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"] },
  { name: "RIGHT_SHOULDER", label: "right shoulder", points: ["RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"] },
];

/** Calculate all joint angles from a set of keypoints */
function calculateJointAngles(joints: Record<string, { x: number; y: number }>): Record<string, number> {
  const angles: Record<string, number> = {};
  
  for (const { name, points } of JOINT_ANGLES) {
    const [aName, bName, cName] = points;
    const a = joints[aName];
    const b = joints[bName];
    const c = joints[cName];
    
    if (a && b && c) {
      angles[name] = calculateAngle(a, b, c);
    }
  }
  
  return angles;
}

function calculateJointConfidence(
  keypointScores: Record<string, number>
): Record<string, number> {
  const conf: Record<string, number> = {};
  for (const { name, points } of JOINT_ANGLES) {
    const [a, b, c] = points;
    const ca = keypointScores[a];
    const cb = keypointScores[b];
    const cc = keypointScores[c];
    if (ca === undefined || cb === undefined || cc === undefined) continue;
    conf[name] = Math.min(ca, cb, cc);
  }
  return conf;
}

/** Calculate median of an array of numbers */
function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Get median angles from a buffer of angle records */
function getMedianAngles(buffer: Record<string, number>[]): Record<string, number> {
  if (buffer.length === 0) return {};
  
  const result: Record<string, number> = {};
  
  for (const { name } of JOINT_ANGLES) {
    const values = buffer
      .map(frame => frame[name])
      .filter((v): v is number => v !== undefined);
    
    if (values.length >= 3) { // Require at least 3 samples for a stable median
      result[name] = median(values);
    }
  }
  
  return result;
}

function getMedianConfidence(buffer: Record<string, number>[]): Record<string, number> {
  if (buffer.length === 0) return {};
  const result: Record<string, number> = {};
  for (const { name } of JOINT_ANGLES) {
    const values = buffer
      .map((frame) => frame[name])
      .filter((v): v is number => v !== undefined);
    if (values.length >= 3) {
      result[name] = median(values);
    }
  }
  return result;
}

/** Compare two sets of joint angles and return score + worst joint */
function comparePoses(
  userAngles: Record<string, number>,
  refAngles: Record<string, number>
): { score: number; worstJoint: string | null; worstDiff: number; jointsCompared: number } {
  const diffs: { name: string; label: string; diff: number }[] = [];

  for (const { name, label } of JOINT_ANGLES) {
    const userAngle = userAngles[name];
    const refAngle = refAngles[name];

    if (userAngle !== undefined && refAngle !== undefined) {
      diffs.push({ name, label, diff: Math.abs(userAngle - refAngle) });
    }
  }

  if (diffs.length === 0) {
    return { score: -1, worstJoint: null, worstDiff: 0, jointsCompared: 0 };
  }

  // Make the score robust to a single outlier joint by using a trimmed mean:
  // sort diffs and drop the single worst joint if we have at least 3.
  const sortedDiffs = [...diffs].sort((a, b) => a.diff - b.diff);
  const usedDiffs =
    sortedDiffs.length >= 3 ? sortedDiffs.slice(0, sortedDiffs.length - 1) : sortedDiffs;

  const avgDiff =
    usedDiffs.reduce((sum, d) => sum + d.diff, 0) / usedDiffs.length;
  const score = Math.max(0, Math.min(100, 100 - avgDiff));

  const worst = diffs.reduce((max, d) => (d.diff > max.diff ? d : max), diffs[0]);

  return {
    score,
    worstJoint: worst.diff > 15 ? worst.label : null,
    worstDiff: worst.diff,
    jointsCompared: usedDiffs.length,
  };
}

function msToTimestamp(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

const CHUNK_MIN_MS = 4000;
const CHUNK_MAX_MS = 6000;
/** Full-phrase learn drill evaluation only (chunk drill no longer gates on score). */
const PHRASE_EVAL_THRESHOLD_DEG = 25;
/** Countdown before the full-phrase run starts. */
const PHRASE_PREP_COUNTDOWN_SEC = 5;

/** Key poses: low movement-velocity valleys in reference skeleton, ≥0.5s apart. */
const KEY_POSE_MIN_SEPARATION_MS = 500;
const KEY_POSE_COUNT_MIN = 2;
const KEY_POSE_COUNT_MAX = 3;

/** MoveNet collapses — treat as tracking garbage (user readings only). */
const KEY_POSE_USER_ANGLE_MIN_PLAUSIBLE = 8;
const KEY_POSE_USER_ANGLE_MAX_PLAUSIBLE = 172;
const KEY_POSE_PER_JOINT_CONF_MIN = 0.35;
/** Need enough trustworthy joints per reference target after filters; else try next-closest keyframe. */
const KEY_POSE_MIN_VALID_USER_JOINTS = 3;

/** Split a phrase into sub-chunks of roughly 4–6 seconds each. */
function splitSegmentIntoChunks(startMs: number, endMs: number): { startMs: number; endMs: number }[] {
  const dur = endMs - startMs;
  if (dur <= 0) return [];
  const n = Math.max(1, Math.ceil(dur / 5000));
  const chunkDur = Math.min(CHUNK_MAX_MS, Math.max(CHUNK_MIN_MS, Math.floor(dur / n)));
  const chunks: { startMs: number; endMs: number }[] = [];
  let cur = startMs;
  while (cur < endMs) {
    const end = Math.min(cur + chunkDur, endMs);
    const len = end - cur;
    if (len < CHUNK_MIN_MS && chunks.length > 0) {
      chunks[chunks.length - 1].endMs = end;
    } else {
      chunks.push({ startMs: cur, endMs: end });
    }
    cur = end;
  }
  return chunks;
}

/** Median joint angles from skeleton frames in [startMs, endMs]. */
function medianAnglesForSkeletonWindow(
  frames: SkeletonFrame[],
  startMs: number,
  endMs: number
): Record<string, number> {
  const samples: Record<string, number>[] = [];
  for (const f of frames) {
    const t = f.timestamp_ms ?? 0;
    if (t < startMs || t > endMs) continue;
    const lm = f.landmarks ?? [];
    if (lm.length === 0) continue;
    const angles = calculateJointAngles(landmarkMap(lm));
    if (Object.keys(angles).length > 0) samples.push(angles);
  }
  return getMedianAngles(samples);
}

/** Interpolate reference joint angles at an exact timestamp (linear interpolation). */
function interpolatedAnglesAtTime(
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
  const la = calculateJointAngles(landmarkMap(l.landmarks ?? []));
  const ra = calculateJointAngles(landmarkMap(r.landmarks ?? []));
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

/** Per-joint signed/absolute diffs for chunk-drill logging (same joints as compareChunkToReference). */
function computeChunkJointDiffsForLog(
  userMed: Record<string, number>,
  refMed: Record<string, number>
): Record<string, { userDeg: number; refDeg: number; signedDiff: number; absDiff: number }> {
  const out: Record<string, { userDeg: number; refDeg: number; signedDiff: number; absDiff: number }> = {};
  for (const { name } of JOINT_ANGLES) {
    const u = userMed[name];
    const r = refMed[name];
    if (u === undefined || r === undefined) continue;
    const signed = u - r;
    out[name] = { userDeg: u, refDeg: r, signedDiff: signed, absDiff: Math.abs(signed) };
  }
  return out;
}

function compareChunkToReference(
  userMed: Record<string, number>,
  userMedConf: Record<string, number>,
  refMed: Record<string, number>,
  thresholdDeg: number
): { pass: true } | { pass: false; worstKey: string; worstLabel: string; worstDiff: number } {
  const diffs: { name: string; label: string; diff: number; conf: number }[] = [];
  for (const { name, label } of JOINT_ANGLES) {
    const u = userMed[name];
    const r = refMed[name];
    if (u === undefined || r === undefined) continue;
    diffs.push({ name, label, diff: Math.abs(u - r), conf: userMedConf[name] ?? 0 });
  }
  if (diffs.length === 0) return { pass: true };

  // Prefer reliable joints for coaching focus in chunk drill.
  const RELIABLE_CONF_MIN = 0.45;
  const reliable = diffs.filter((d) => d.conf >= RELIABLE_CONF_MIN);
  const pool = reliable.length > 0 ? reliable : diffs;
  const sorted = [...pool].sort((a, b) => b.diff - a.diff);
  const focus = sorted[0];

  if (!focus || focus.diff <= thresholdDeg) return { pass: true };
  return { pass: false, worstKey: focus.name, worstLabel: focus.label, worstDiff: focus.diff };
}

/** Largest joint vs reference diff for yellow highlight — no pass/fail threshold. */
function pickHighlightJointForChunkFeedback(
  userMed: Record<string, number>,
  userMedConf: Record<string, number>,
  refMed: Record<string, number>
): { worstKey: string; worstLabel: string; worstDiff: number } | null {
  const diffs: { name: string; label: string; diff: number; conf: number }[] = [];
  for (const { name, label } of JOINT_ANGLES) {
    const u = userMed[name];
    const r = refMed[name];
    if (u === undefined || r === undefined) continue;
    diffs.push({ name, label, diff: Math.abs(u - r), conf: userMedConf[name] ?? 0 });
  }
  if (diffs.length === 0) return null;
  const RELIABLE_CONF_MIN = 0.45;
  const reliable = diffs.filter((d) => d.conf >= RELIABLE_CONF_MIN);
  const pool = reliable.length > 0 ? reliable : diffs;
  const sorted = [...pool].sort((a, b) => b.diff - a.diff);
  const focus = sorted[0]!;
  return { worstKey: focus.name, worstLabel: focus.label, worstDiff: focus.diff };
}

/**
 * Max abs diff per joint across key-pose samples (mirrors former key-pose aggregation; no scoring gate).
 */
function pickHighlightJointFromKeyPosePoses(
  poses: { user: Record<string, number>; ref: Record<string, number>; conf: Record<string, number> }[]
): { worstKey: string; worstLabel: string; worstDiff: number } | null {
  const jointAgg: { name: string; label: string; maxDiff: number; minConf: number }[] = [];
  for (const { name, label } of JOINT_ANGLES) {
    let maxDiff = 0;
    let minConf = 1;
    let any = false;
    for (const p of poses) {
      const u = p.user[name];
      const r = p.ref[name];
      if (u === undefined || r === undefined) continue;
      if (!userJointPassesKeyPoseTrackingFilters(name, p.user, p.conf)) continue;
      any = true;
      const d = Math.abs(u - r);
      if (d > maxDiff) maxDiff = d;
      const c = p.conf[name] ?? 0;
      if (c < minConf) minConf = c;
    }
    if (any) jointAgg.push({ name, label, maxDiff, minConf });
  }
  if (jointAgg.length === 0) return null;
  const RELIABLE_CONF_MIN = 0.45;
  const reliable = jointAgg.filter((j) => j.minConf >= RELIABLE_CONF_MIN);
  const pool = reliable.length > 0 ? reliable : jointAgg;
  const sorted = [...pool].sort((a, b) => b.maxDiff - a.maxDiff);
  const focus = sorted[0]!;
  return { worstKey: focus.name, worstLabel: focus.label, worstDiff: focus.maxDiff };
}

function averageJointDiff(
  userAngles: Record<string, number>,
  refAngles: Record<string, number>
): { avgDiff: number; jointsCompared: number } {
  let sum = 0;
  let count = 0;
  for (const { name } of JOINT_ANGLES) {
    const u = userAngles[name];
    const r = refAngles[name];
    if (u === undefined || r === undefined) continue;
    sum += Math.abs(u - r);
    count += 1;
  }
  return { avgDiff: count > 0 ? sum / count : Number.POSITIVE_INFINITY, jointsCompared: count };
}

/** Sum of Euclidean distances per landmark name between consecutive frames (x,y plane). */
function sumLandmarkDisplacement(prev: Landmark[], curr: Landmark[]): number {
  if (prev.length === 0 || curr.length === 0) return 0;
  const currByName = landmarkMap(curr);
  let sum = 0;
  let n = 0;
  for (const p of prev) {
    const q = currByName[p.name];
    if (!q) continue;
    const dx = p.x - q.x;
    const dy = p.y - q.y;
    sum += Math.hypot(dx, dy);
    n += 1;
  }
  return n > 0 ? sum : 0;
}

function getSortedChunkSkeletonFrames(
  frames: SkeletonFrame[],
  startMs: number,
  endMs: number
): SkeletonFrame[] {
  const inWin = frames.filter((f) => {
    const t = f.timestamp_ms ?? 0;
    return t >= startMs && t <= endMs;
  });
  inWin.sort((a, b) => (a.timestamp_ms ?? 0) - (b.timestamp_ms ?? 0));
  return inWin;
}

/** velocities[i] = displacement from frames[i-1] → frames[i]; velocities[0] = 0. */
function movementVelocitySeries(frames: SkeletonFrame[]): number[] {
  const v: number[] = [0];
  for (let i = 1; i < frames.length; i++) {
    v.push(sumLandmarkDisplacement(frames[i - 1]?.landmarks ?? [], frames[i]?.landmarks ?? []));
  }
  return v;
}

function findMovementVelocityLocalMinimaIndices(velocities: number[]): number[] {
  const out: number[] = [];
  if (velocities.length < 4) return out;
  for (let i = 2; i < velocities.length - 1; i++) {
    const vi = velocities[i]!;
    if (vi < velocities[i - 1]! && vi < velocities[i + 1]!) out.push(i);
  }
  return out;
}

function greedyPickKeyPoseIndicesByLowVelocity(
  frames: SkeletonFrame[],
  velocities: number[],
  candidateIndices: number[],
  minSepMs: number,
  maxCount: number
): number[] {
  const sorted = [...candidateIndices].sort((a, b) => (velocities[a] ?? 0) - (velocities[b] ?? 0));
  const picked: number[] = [];
  for (const i of sorted) {
    const ti = frames[i]?.timestamp_ms ?? 0;
    if (picked.some((pi) => Math.abs(ti - (frames[pi]?.timestamp_ms ?? 0)) < minSepMs)) continue;
    picked.push(i);
    if (picked.length >= maxCount) break;
  }
  return picked;
}

function selectKeyPoseFrameIndices(
  frames: SkeletonFrame[],
  minSepMs: number,
  minCount: number,
  maxCount: number
): { indices: number[]; velocities: number[]; source: "local_min" | "fallback_lowest" } {
  const velocities = movementVelocitySeries(frames);
  if (frames.length < 2) return { indices: [], velocities, source: "local_min" };

  const localMinIdx = findMovementVelocityLocalMinimaIndices(velocities);
  let picked = greedyPickKeyPoseIndicesByLowVelocity(frames, velocities, localMinIdx, minSepMs, maxCount);
  let source: "local_min" | "fallback_lowest" = "local_min";

  if (picked.length < minCount) {
    const allMotion = Array.from({ length: frames.length - 1 }, (_, j) => j + 1);
    picked = greedyPickKeyPoseIndicesByLowVelocity(frames, velocities, allMotion, minSepMs, maxCount);
    source = "fallback_lowest";
  }

  picked.sort((a, b) => (frames[a]?.timestamp_ms ?? 0) - (frames[b]?.timestamp_ms ?? 0));
  return { indices: picked, velocities, source };
}

function averageJointDiffAcrossKeyPoses(
  pairs: { user: Record<string, number>; ref: Record<string, number> }[]
): { avgDiff: number; jointsCompared: number } {
  if (pairs.length === 0) return { avgDiff: Number.POSITIVE_INFINITY, jointsCompared: 0 };
  let sum = 0;
  let n = 0;
  for (const p of pairs) {
    const d = averageJointDiff(p.user, p.ref);
    if (d.jointsCompared > 0) {
      sum += d.avgDiff;
      n += 1;
    }
  }
  return { avgDiff: n > 0 ? sum / n : Number.POSITIVE_INFINITY, jointsCompared: n };
}

/** Mean per joint across key-pose samples (for coaching payload). */
function userJointPassesKeyPoseTrackingFilters(
  jointName: string,
  userAngles: Record<string, number>,
  conf: Record<string, number>
): boolean {
  const u = userAngles[jointName];
  if (u === undefined) return false;
  if (u < KEY_POSE_USER_ANGLE_MIN_PLAUSIBLE || u > KEY_POSE_USER_ANGLE_MAX_PLAUSIBLE) return false;
  if ((conf[jointName] ?? 0) < KEY_POSE_PER_JOINT_CONF_MIN) return false;
  return true;
}

/** Drop joints that fail tracking filters so mirror averaging matches key-pose aggregation. */
function filterUserAnglesForKeyPoseMirrorCompare(
  userAngles: Record<string, number>,
  conf: Record<string, number>
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const { name } of JOINT_ANGLES) {
    if (!userJointPassesKeyPoseTrackingFilters(name, userAngles, conf)) continue;
    const u = userAngles[name];
    if (u !== undefined) out[name] = u;
  }
  return out;
}

function countUserJointsAfterKeyPoseFilters(
  userAngles: Record<string, number>,
  conf: Record<string, number>
): {
  valid: number;
  filteredExtremeAngle: number;
  filteredLowConfidence: number;
  missingAngle: number;
} {
  let valid = 0;
  let filteredExtremeAngle = 0;
  let filteredLowConfidence = 0;
  let missingAngle = 0;
  for (const { name } of JOINT_ANGLES) {
    const u = userAngles[name];
    if (u === undefined) {
      missingAngle += 1;
      continue;
    }
    if (u < KEY_POSE_USER_ANGLE_MIN_PLAUSIBLE || u > KEY_POSE_USER_ANGLE_MAX_PLAUSIBLE) {
      filteredExtremeAngle += 1;
      continue;
    }
    if ((conf[name] ?? 0) < KEY_POSE_PER_JOINT_CONF_MIN) {
      filteredLowConfidence += 1;
      continue;
    }
    valid += 1;
  }
  return { valid, filteredExtremeAngle, filteredLowConfidence, missingAngle };
}

function attemptKeyframeIndicesSortedByTimeProximity(
  attemptKfs: RecordedAttemptKeyframe[],
  targetMs: number
): number[] {
  return attemptKfs
    .map((kf, i) => ({ i, d: Math.abs(kf.timestamp_ms - targetMs) }))
    .sort((a, b) => a.d - b.d)
    .map((x) => x.i);
}

type KeyPosePickAttemptLog = {
  keyframeIndex: number;
  keyframeTimestampMs: number;
  deltaMsFromTarget: number;
  validJointCount: number;
  filteredExtremeAngle: number;
  filteredLowConfidence: number;
  missingAngle: number;
  rejectedTooFewJoints: boolean;
};

type KeyPosePickResult = {
  keyframeIndex: number;
  user: Record<string, number>;
  conf: Record<string, number>;
  attempts: KeyPosePickAttemptLog[];
};

/**
 * For each reference time, try closest user keyframes until one has ≥ KEY_POSE_MIN_VALID_USER_JOINTS
 * joints passing angle + confidence filters.
 */
function pickUserKeyframeForReferenceKeyPose(
  attemptKfs: RecordedAttemptKeyframe[],
  confBuf: Record<string, number>[],
  referenceTargetMs: number
): KeyPosePickResult | null {
  const ordered = attemptKeyframeIndicesSortedByTimeProximity(attemptKfs, referenceTargetMs);
  const attempts: KeyPosePickAttemptLog[] = [];

  for (const ik of ordered) {
    const kf = attemptKfs[ik];
    if (!kf) continue;
    const user = calculateJointAngles(kf.joints);
    const conf = ik >= 0 && ik < confBuf.length ? confBuf[ik]! : {};
    const counts = countUserJointsAfterKeyPoseFilters(user, conf);
    const rejected = counts.valid < KEY_POSE_MIN_VALID_USER_JOINTS;
    attempts.push({
      keyframeIndex: ik,
      keyframeTimestampMs: kf.timestamp_ms,
      deltaMsFromTarget: Math.abs(kf.timestamp_ms - referenceTargetMs),
      validJointCount: counts.valid,
      filteredExtremeAngle: counts.filteredExtremeAngle,
      filteredLowConfidence: counts.filteredLowConfidence,
      missingAngle: counts.missingAngle,
      rejectedTooFewJoints: rejected,
    });
    if (!rejected) {
      return { keyframeIndex: ik, user, conf, attempts };
    }
  }
  return null;
}

function meanJointAnglesAcrossSamples(samples: Record<string, number>[]): Record<string, number> {
  const acc: Record<string, number[]> = {};
  for (const s of samples) {
    for (const [k, v] of Object.entries(s)) {
      if (!acc[k]) acc[k] = [];
      acc[k]!.push(v);
    }
  }
  const out: Record<string, number> = {};
  for (const { name } of JOINT_ANGLES) {
    const arr = acc[name];
    if (arr && arr.length > 0) {
      out[name] = arr.reduce((s, x) => s + x, 0) / arr.length;
    }
  }
  return out;
}

type LearnDrillStep =
  | "preview"
  | "attempt"
  | "chunk_feedback"
  | "phrase_prep"
  | "put_together"
  | "phrase_review"
  | "phrase_complete";

interface LearnDrillState {
  segmentId: number;
  segment: Segment;
  chunks: { startMs: number; endMs: number }[];
  chunkIndex: number;
  step: LearnDrillStep;
  /** Completed attempts on the current sub-chunk (for optional retry cap). */
  chunkAttemptsOnSlice?: number;
}

interface DanceLogEntry {
  segment_id: number;
  timestamp_ms: number;
  per_joint_abs_diff: Record<string, number>;
}

interface TestDiagnostics {
  score: number;
  jointsCompared: number;
  worstJoint: string | null;
  worstDiff: number;
  validJoints: number;
  avgConfidence: number;
  frameOffsetMs: number;
  mappingMode: "direct" | "lr_swapped";
}

type SessionMirrorMapping = "pending" | "direct" | "lr_swapped";

function swapLeftRightJointAngles(angles: Record<string, number>): Record<string, number> {
  const swapped: Record<string, number> = {};
  for (const { name } of JOINT_ANGLES) {
    if (name.startsWith("LEFT_")) {
      const rhs = `RIGHT_${name.slice(5)}`;
      swapped[name] = angles[rhs] ?? angles[name];
    } else if (name.startsWith("RIGHT_")) {
      const lhs = `LEFT_${name.slice(6)}`;
      swapped[name] = angles[lhs] ?? angles[name];
    } else {
      swapped[name] = angles[name];
    }
  }
  return swapped;
}

function QualityBadge({ score }: { score: number }) {
  const label = (score * 100).toFixed(0) + "%";
  if (score > 0.82) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/20 px-3 py-1 text-sm font-medium text-emerald-400">
        <span className="size-2 rounded-full bg-emerald-400" />
        Quality {label}
      </span>
    );
  }
  if (score >= 0.75) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-500/20 px-3 py-1 text-sm font-medium text-amber-400">
        <span className="size-2 rounded-full bg-amber-400" />
        Quality {label}
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-red-500/20 px-3 py-1 text-sm font-medium text-red-400">
      <span className="size-2 rounded-full bg-red-400" />
      Quality {label}
    </span>
  );
}

/** Draw skeleton on canvas */
function drawSkeleton(
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

function getJointAngleDef(jointName: string) {
  return JOINT_ANGLES.find((j) => j.name === jointName);
}

/** Visual arc at vertex B between BA and BC, with degree label */
function drawJointAngleArc(
  ctx: CanvasRenderingContext2D,
  a: { x: number; y: number },
  b: { x: number; y: number },
  c: { x: number; y: number },
  angleDeg: number,
  color: string
) {
  const ang1 = Math.atan2(a.y - b.y, a.x - b.x);
  const ang2 = Math.atan2(c.y - b.y, c.x - b.x);
  let sweep = ang2 - ang1;
  while (sweep > Math.PI) sweep -= 2 * Math.PI;
  while (sweep < -Math.PI) sweep += 2 * Math.PI;

  const leg1 = Math.hypot(a.x - b.x, a.y - b.y);
  const leg2 = Math.hypot(c.x - b.x, c.y - b.y);
  const r = Math.min(52, leg1 * 0.28, leg2 * 0.28);

  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.arc(b.x, b.y, r, ang1, ang1 + sweep, sweep < 0);
  ctx.stroke();

  const mid = ang1 + sweep / 2;
  const lx = b.x + Math.cos(mid) * (r + 16);
  const ly = b.y + Math.sin(mid) * (r + 16);
  ctx.fillStyle = color;
  ctx.font = "bold 11px ui-monospace, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(`${angleDeg.toFixed(0)}°`, lx, ly);
}

export default function PlayerPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const videoId = params?.videoId as string | undefined;
  const jobId = searchParams?.get("jobId") ?? null;

  const [choreoData, setChoreoData] = useState<ChoreoData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [playbackRate, setPlaybackRate] = useState<1 | 0.75 | 0.5>(1);
  const [isReferenceMirrored, setIsReferenceMirrored] = useState(false);
  const [activeSegmentId, setActiveSegmentId] = useState<number | null>(null);
  const [showSkeleton, setShowSkeleton] = useState(false);
  const [skeletonFrames, setSkeletonFrames] = useState<SkeletonFrame[]>([]);
  const [skeletonLoading, setSkeletonLoading] = useState(false);

  // Supabase auth (for the user menu)
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [signOutLoading, setSignOutLoading] = useState(false);

  // Webcam state
  const [practiceMode, setPracticeMode] = useState(false);
  const [webcamLoading, setWebcamLoading] = useState(false);
  const [webcamError, setWebcamError] = useState<string | null>(null);
  const [detectorReady, setDetectorReady] = useState(false);
  const [useTestVideo, setUseTestVideo] = useState(false); // Use reference video as input for testing

  // Pose comparison state
  const [poseScore, setPoseScore] = useState<number | null>(null);
  const [worstJoint, setWorstJoint] = useState<string | null>(null);
  const [displayedScore, setDisplayedScore] = useState<number | null>(null); // Smoothed score (internal only)
  const [sessionMirrorMapping, setSessionMirrorMapping] = useState<SessionMirrorMapping>("pending");
  const [completedLearnDrillSegmentId, setCompletedLearnDrillSegmentId] = useState<number | null>(null);
  const [fullPhraseEvalResult, setFullPhraseEvalResult] = useState<{ pass: boolean; note: string } | null>(null);
  const sessionMirrorMappingRef = useRef<SessionMirrorMapping>("pending");
  sessionMirrorMappingRef.current = sessionMirrorMapping;

  // Coaching state
  const [coachingNote, setCoachingNote] = useState<string | null>(null);
  const [coachingLoading, setCoachingLoading] = useState(false);

  // Positioning gate (preflight) before starting practice
  const [positioningActive, setPositioningActive] = useState(false);
  const [positioningWarning, setPositioningWarning] = useState<string | null>(null);

  // Coaching text-to-speech
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [speechSupported, setSpeechSupported] = useState(false);
  const ttsVoiceRef = useRef<SpeechSynthesisVoice | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const loopIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const videoContainerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Keep the top-bar user menu in sync with auth state.
  useEffect(() => {
    let mounted = true;

    async function load() {
      setAuthLoading(true);
      try {
        const { data, error } = await supabase.auth.getUser();
        if (!mounted) return;
        if (error) setUserEmail(null);
        else setUserEmail(data.user?.email ?? null);
      } catch {
        if (!mounted) return;
        setUserEmail(null);
      } finally {
        if (!mounted) return;
        setAuthLoading(false);
      }
    }

    void load();

    const { data: sub } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setUserEmail(session?.user?.email ?? null);
        setAuthLoading(false);
      }
    );

    return () => {
      mounted = false;
      sub.subscription.unsubscribe();
    };
  }, []);

  const handleSignOut = useCallback(async () => {
    setSignOutLoading(true);
    try {
      await supabase.auth.signOut();
      router.replace("/auth/login");
    } finally {
      setSignOutLoading(false);
    }
  }, [router]);

  // Webcam refs
  const webcamRef = useRef<HTMLVideoElement | null>(null);
  const webcamCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const webcamContainerRef = useRef<HTMLDivElement | null>(null);
  const webcamStreamRef = useRef<MediaStream | null>(null);
  const detectorRef = useRef<unknown>(null);
  const webcamRafRef = useRef<number | null>(null);
  const latestUserJointsRef = useRef<Record<string, { x: number; y: number }> | null>(null);
  const positioningOkRef = useRef(false);
  
  // Rolling buffer for pose smoothing (stores joint angles, not raw keypoints)
  const jointAnglesBufferRef = useRef<Record<string, number>[]>([]);
  const jointConfidenceBufferRef = useRef<Record<string, number>[]>([]);
  const BUFFER_SIZE = 30;

  // Coaching timing/throttling
  const segmentSecondsRef = useRef<number>(0);
  const lastCoachingTimeRef = useRef<number>(0);
  const lostBodyCountRef = useRef<number>(0);

  // Learn vs Dance mode: Learn = slow, detailed; Dance = full speed, short cues
  const [practiceModeType, setPracticeModeType] = useState<"learn" | "dance">("learn");
  const [suggestDanceMode, setSuggestDanceMode] = useState(false);
  const learnModeSecondsRef = useRef<number>(0);

  // Learn mode: pause reference video when new coaching appears so user can read it
  const [learnPausedForCoaching, setLearnPausedForCoaching] = useState(false);
  const learnResumeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /** Chunk drill (Learn mode, webcam only) */
  const [learnDrill, setLearnDrill] = useState<LearnDrillState | null>(null);
  const learnDrillRef = useRef<LearnDrillState | null>(null);
  learnDrillRef.current = learnDrill;
  const learnAttemptBufferRef = useRef<Record<string, number>[]>([]);
  const learnAttemptConfidenceBufferRef = useRef<Record<string, number>[]>([]);
  /** Raw MoveNet joints + capture timestamp per inference frame during chunk attempts. */
  const learnAttemptKeypointsBufferRef = useRef<RecordedAttemptKeyframe[]>([]);
  const putTogetherBufferRef = useRef<Record<string, number>[]>([]);
  /** Keyframed webcam poses during full-phrase run (Learn put_together). */
  const putTogetherKeyframesRef = useRef<RecordedAttemptKeyframe[]>([]);
  const putTogetherSmoothedRef = useRef<Record<string, { x: number; y: number }> | null>(null);
  const learnCaptureRef = useRef<"none" | "chunk_attempt" | "put_together">("none");
  /** Last chunk feedback run key `${chunkIndex}-${attemptOnSlice}` — avoids duplicate processing. */
  const chunkFeedbackProcessedKeyRef = useRef<string>("");
  const putTogetherStartedRef = useRef(false);
  const queuedAttemptStartRef = useRef(false);

  /** Dance mode: silent performance log */
  const dancePerformanceLogRef = useRef<DanceLogEntry[]>([]);
  const [performanceReview, setPerformanceReview] = useState<{
    overall_assessment: string;
    top_phrases: { segment_id?: number; reason?: string }[];
    next_session_focus: string;
  } | null>(null);
  const [performanceReviewLoading, setPerformanceReviewLoading] = useState(false);
  const [attemptCountdownSec, setAttemptCountdownSec] = useState<number | null>(null);
  const [testDiagnostics, setTestDiagnostics] = useState<TestDiagnostics | null>(null);
  /** 3 → 2 → 1 before chunk attempt recording starts (after Start your turn / gate pass). */
  const [preAttemptCountdownSec, setPreAttemptCountdownSec] = useState<number | null>(null);
  const preAttemptCountdownRef = useRef<number | null>(null);
  preAttemptCountdownRef.current = preAttemptCountdownSec;
  /** 5 → ... → 1 before full-phrase "put it together" starts after the last chunk. */
  const [phrasePrepCountdownSec, setPhrasePrepCountdownSec] = useState<number | null>(null);

  /** Keyframes copied when entering phrase_review (Learn full phrase). */
  const [phraseReviewLearnKeyframes, setPhraseReviewLearnKeyframes] = useState<RecordedAttemptKeyframe[]>([]);

  /** Dance mode: keyed poses for whichever phrase window the video is playing. */
  const dancePhraseKeyframesRef = useRef<RecordedAttemptKeyframe[]>([]);
  const dancePhraseSmoothedRef = useRef<Record<string, { x: number; y: number }> | null>(null);
  const [phraseReviewDance, setPhraseReviewDance] = useState<{
    segment: Segment;
    keyframes: RecordedAttemptKeyframe[];
  } | null>(null);
  const phraseReviewDanceRef = useRef<{ segment: Segment; keyframes: RecordedAttemptKeyframe[] } | null>(null);
  phraseReviewDanceRef.current = phraseReviewDance;

  /** Saved keyframes from the last chunk attempt — used for synced post-attempt review playback. */
  const [attemptReviewKeypoints, setAttemptReviewKeypoints] = useState<RecordedAttemptKeyframe[]>([]);
  /** Ref mirrors state so chunk feedback runs with the keyframes from the attempt that just ended. */
  const attemptReviewKeypointsRef = useRef<RecordedAttemptKeyframe[]>([]);
  attemptReviewKeypointsRef.current = attemptReviewKeypoints;
  const [chunkReviewPlayback, setChunkReviewPlayback] = useState(false);
  const chunkReviewPlaybackRef = useRef(false);
  const replayJointsRef = useRef<Record<string, { x: number; y: number }> | null>(null);
  const prevAttemptSmoothedJointsRef = useRef<Record<string, { x: number; y: number }> | null>(null);
  chunkReviewPlaybackRef.current = chunkReviewPlayback;

  /** Hide floating Learn drill card during attempt replay so videos stay unobstructed; reopen via bottom bar. */
  const [learnFeedbackPanelVisible, setLearnFeedbackPanelVisible] = useState(true);

  useEffect(() => {
    if (chunkReviewPlayback) {
      setLearnFeedbackPanelVisible(false);
    } else {
      setLearnFeedbackPanelVisible(true);
    }
  }, [chunkReviewPlayback]);

  /** Attempt-review debug: live scrub time + worst joint from last chunk evaluation */
  const [reviewPlaybackTimeMs, setReviewPlaybackTimeMs] = useState(0);
  const [chunkDrillFocusJoint, setChunkDrillFocusJoint] = useState<string | null>(null);
  const chunkDrillFocusJointRef = useRef<string | null>(null);
  chunkDrillFocusJointRef.current = chunkDrillFocusJoint;

  const [debugMode, setDebugMode] = useState(() => process.env.NODE_ENV === "development");
  const debugModeRef = useRef(debugMode);
  debugModeRef.current = debugMode;

  const segments = choreoData?.segments ?? [];
  const overallScore = choreoData?.quality_assessment?.overall_score ?? 0;

  // Fetch choreo_data by jobId
  useEffect(() => {
    if (!jobId || !videoId) {
      setError("Missing job or video. Start from the homepage.");
      setLoading(false);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_URL}/status/${jobId}`);
        const data = await res.json();
        if (cancelled) return;
        if (data.status !== "complete") {
          setError(data.status === "failed" ? data.error ?? "Job failed" : "Analysis not ready.");
          setLoading(false);
          return;
        }
        setChoreoData(data.choreo_data ?? null);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to load");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [jobId, videoId]);

  // Reminder: this page is a client component — console.log goes to the browser, not the terminal
  useEffect(() => {
    if (loading || error || !choreoData) return;
    console.info(
      "%c[chunk-drill] Player loaded. All [chunk-drill] messages go to the BROWSER console (Inspect → Console), not your Next.js terminal.",
      "color:#a1a1aa"
    );
  }, [loading, error, choreoData]);

  // Fetch skeleton when user enables overlay
  useEffect(() => {
    if (!showSkeleton || !videoId || skeletonFrames.length > 0) return;
    setSkeletonLoading(true);
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_URL}/skeleton/${videoId}`);
        if (!res.ok) {
          if (!cancelled) setSkeletonLoading(false);
          return;
        }
        const data: SkeletonFrame[] = await res.json();
        if (!cancelled) {
          setSkeletonFrames(Array.isArray(data) ? data : []);
          setSkeletonLoading(false);
        }
      } catch {
        if (!cancelled) setSkeletonLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [showSkeleton, videoId, skeletonFrames.length]);

  const stopLoop = useCallback(() => {
    if (loopIntervalRef.current) {
      clearInterval(loopIntervalRef.current);
      loopIntervalRef.current = null;
    }
    setActiveSegmentId(null);
    setLearnDrill(null);
    learnDrillRef.current = null;
    learnCaptureRef.current = "none";
    learnAttemptBufferRef.current = [];
    learnAttemptConfidenceBufferRef.current = [];
    learnAttemptKeypointsBufferRef.current = [];
    prevAttemptSmoothedJointsRef.current = null;
    setAttemptReviewKeypoints([]);
    setChunkReviewPlayback(false);
    replayJointsRef.current = null;
    setChunkDrillFocusJoint(null);
    setReviewPlaybackTimeMs(0);
    setCompletedLearnDrillSegmentId(null);
    setFullPhraseEvalResult(null);
    chunkFeedbackProcessedKeyRef.current = "";
    queuedAttemptStartRef.current = false;
    setPreAttemptCountdownSec(null);
    setPhrasePrepCountdownSec(null);
    putTogetherKeyframesRef.current = [];
    putTogetherSmoothedRef.current = null;
    setPhraseReviewLearnKeyframes([]);
    dancePhraseKeyframesRef.current = [];
    dancePhraseSmoothedRef.current = null;
    setPhraseReviewDance(null);
  }, []);

  const playSegment = useCallback(
    (seg: Segment) => {
      const video = videoRef.current;
      if (!video) return;

      // Chunk drill only runs after user explicitly starts practice mode.
      const willChunkDrill = practiceMode && !useTestVideo && practiceModeType === "learn";
      console.log("[chunk-drill] playSegment()", {
        segmentId: seg.segment_id,
        practiceMode,
        useTestVideo,
        practiceModeType,
        willChunkDrill,
      });
      if (!willChunkDrill) {
        console.log("[chunk-drill] chunk drill OFF — need: practice ON, Learn tab selected, and NOT Test Mode", {
          practiceMode,
          useTestVideo,
          practiceModeType,
        });
      }

      setActiveSegmentId(seg.segment_id);
      setCompletedLearnDrillSegmentId(null);
      setFullPhraseEvalResult(null);
      setCoachingNote(null);
      setSuggestDanceMode(false);
      learnModeSecondsRef.current = 0;

      // Learn: chunk drill (preview → attempt → feedback per sub-chunk)
      if (willChunkDrill) {
        setShowSkeleton(true);
        const chunks = splitSegmentIntoChunks(seg.start_time_ms, seg.end_time_ms);
        const newDrill: LearnDrillState = {
          segmentId: seg.segment_id,
          segment: seg,
          chunks,
          chunkIndex: 0,
          step: "preview",
          chunkAttemptsOnSlice: 0,
        };
        learnDrillRef.current = newDrill;
        setLearnDrill(newDrill);
        learnCaptureRef.current = "none";
        learnAttemptBufferRef.current = [];
        learnAttemptConfidenceBufferRef.current = [];
        learnAttemptKeypointsBufferRef.current = [];
        prevAttemptSmoothedJointsRef.current = null;
        setAttemptReviewKeypoints([]);
        setChunkReviewPlayback(false);
        setChunkDrillFocusJoint(null);
        setReviewPlaybackTimeMs(0);
        chunkFeedbackProcessedKeyRef.current = "";
        setPreAttemptCountdownSec(null);
        setPhrasePrepCountdownSec(null);
        setPlaybackRate(0.5);
        video.playbackRate = 0.5;
        const c0 = chunks[0];
        video.currentTime = c0.startMs / 1000;
        video.play();
        console.log("[chunk-drill] STATE → drill session started (preview)", {
          segmentId: seg.segment_id,
          chunkCount: chunks.length,
          chunks: chunks.map((ch, i) => ({ i, startMs: ch.startMs, endMs: ch.endMs })),
        });
        return;
      }

      setLearnDrill(null);
      learnDrillRef.current = null;
      learnCaptureRef.current = "none";

      const startSec = seg.start_time_ms / 1000;
      const endSec = seg.end_time_ms / 1000;
      const rate = practiceModeType === "learn" ? 0.5 : 1;
      setPlaybackRate(rate as 1 | 0.75 | 0.5);
      video.playbackRate = rate;
      video.currentTime = startSec;
      video.play();

      if (loopIntervalRef.current) clearInterval(loopIntervalRef.current);
      loopIntervalRef.current = setInterval(() => {
        const v = videoRef.current;
        if (!v) return;
        if (v.currentTime >= endSec - 0.1) {
          v.currentTime = startSec;
        }
      }, 200);
    },
    [practiceMode, useTestVideo, practiceModeType]
  );

  useEffect(() => {
    return () => {
      if (loopIntervalRef.current) clearInterval(loopIntervalRef.current);
    };
  }, []);

  // Learn chunk drill: loop current sub-chunk during preview
  useEffect(() => {
    if (!learnDrill || learnDrill.step !== "preview") return;
    const video = videoRef.current;
    if (!video) return;
    const chunkIndex = learnDrill.chunkIndex;
    const segmentId = learnDrill.segmentId;
    const c = learnDrill.chunks[chunkIndex];
    if (!c) return;
    console.log("[chunk-drill] preview START", {
      chunkIndex,
      startMs: c.startMs,
      endMs: c.endMs,
      startSec: c.startMs / 1000,
      endSec: c.endMs / 1000,
      segmentId,
    });
    video.currentTime = c.startMs / 1000;
    video.playbackRate = 0.5;
    video.play();
    if (loopIntervalRef.current) clearInterval(loopIntervalRef.current);
    loopIntervalRef.current = setInterval(() => {
      const v = videoRef.current;
      const ld = learnDrillRef.current;
      if (!v || !ld || ld.step !== "preview") return;
      const ch = ld.chunks[ld.chunkIndex];
      if (!ch) return;
      if (v.currentTime >= ch.endMs / 1000 - 0.08) {
        v.currentTime = ch.startMs / 1000;
      }
    }, 200);
    return () => {
      console.log("[chunk-drill] preview END", {
        chunkIndex,
        startMs: c.startMs,
        endMs: c.endMs,
        segmentId,
      });
      if (loopIntervalRef.current) {
        clearInterval(loopIntervalRef.current);
        loopIntervalRef.current = null;
      }
    };
  }, [learnDrill?.chunkIndex, learnDrill?.step]);

  /** Post-attempt review: loop the reference chunk at 0.5× and sync saved webcam keyframes to video time */
  useEffect(() => {
    if (!chunkReviewPlayback) {
      replayJointsRef.current = null;
      return;
    }
    const ld = learnDrillRef.current;
    if (!ld || ld.step !== "chunk_feedback") {
      replayJointsRef.current = null;
      return;
    }
    if (attemptReviewKeypoints.length === 0) {
      replayJointsRef.current = null;
      return;
    }
    const video = videoRef.current;
    if (!video) return;

    const c = ld.chunks[ld.chunkIndex];
    if (!c) return;

    const startSec = c.startMs / 1000;
    const endSec = c.endMs / 1000;
    const keyframes = attemptReviewKeypoints;

    video.playbackRate = 0.5;
    video.currentTime = startSec;
    void video.play().catch(() => {
      /* Interrupted by pause() during effect transitions — ignore */
    });

    if (loopIntervalRef.current) {
      clearInterval(loopIntervalRef.current);
      loopIntervalRef.current = null;
    }
    loopIntervalRef.current = setInterval(() => {
      const v = videoRef.current;
      if (!v || !chunkReviewPlaybackRef.current) return;
      if (v.currentTime >= endSec - 0.08) {
        v.currentTime = startSec;
      }
    }, 200);

    const syncReplay = () => {
      const v = videoRef.current;
      if (!v) return;
      const span = endSec - startSec;
      if (span <= 0 || keyframes.length === 0) {
        replayJointsRef.current = null;
        return;
      }
      const targetCaptureMs = v.currentTime * 1000;
      let best = keyframes[0];
      let bestDiff = Math.abs(best.timestamp_ms - targetCaptureMs);
      for (let i = 1; i < keyframes.length; i++) {
        const diff = Math.abs(keyframes[i].timestamp_ms - targetCaptureMs);
        if (diff < bestDiff) {
          best = keyframes[i];
          bestDiff = diff;
        }
      }
      replayJointsRef.current =
        best && Object.keys(best.joints).length > 0 ? { ...best.joints } : null;
      setReviewPlaybackTimeMs(v.currentTime * 1000);
    };

    video.addEventListener("timeupdate", syncReplay);
    video.addEventListener("seeked", syncReplay);
    syncReplay();

    return () => {
      video.removeEventListener("timeupdate", syncReplay);
      video.removeEventListener("seeked", syncReplay);
      replayJointsRef.current = null;
      if (loopIntervalRef.current) {
        clearInterval(loopIntervalRef.current);
        loopIntervalRef.current = null;
      }
      // Only pause when review mode is off. If review stays on but deps changed (e.g. leaving chunk_feedback),
      // pausing here interrupts the in-flight play() and surfaces AbortError in dev.
      if (!chunkReviewPlaybackRef.current) {
        video.pause();
      }
    };
  }, [chunkReviewPlayback, learnDrill?.step, learnDrill?.chunkIndex, attemptReviewKeypoints]);

  // Draw reference skeleton overlay synced to video time
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = videoContainerRef.current;
    const video = videoRef.current;
    if (!canvas || !container || !video || !showSkeleton || skeletonFrames.length === 0) return;

    let rafId: number;

    const draw = () => {
      if (!canvas || !container || !video) return;
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const timeMs = (video.currentTime ?? 0) * 1000;

      let best = 0;
      let bestDiff = Math.abs((skeletonFrames[0]?.timestamp_ms ?? 0) - timeMs);
      for (let i = 1; i < skeletonFrames.length; i++) {
        const diff = Math.abs((skeletonFrames[i]?.timestamp_ms ?? 0) - timeMs);
        if (diff < bestDiff) {
          bestDiff = diff;
          best = i;
        }
      }

      const frame = skeletonFrames[best];
      const landmarks = frame?.landmarks ?? [];
      if (landmarks.length === 0) {
        rafId = requestAnimationFrame(draw);
        return;
      }

      const joints = landmarkMap(landmarks);

      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.scale(dpr, dpr);

      const videoWidth = video.videoWidth || 1;
      const videoHeight = video.videoHeight || 1;
      const containerWidth = rect.width;
      const containerHeight = rect.height;

      const scaleX = containerWidth / videoWidth;
      const scaleY = containerHeight / videoHeight;
      const scale = Math.min(scaleX, scaleY);

      const renderedWidth = videoWidth * scale;
      const renderedHeight = videoHeight * scale;

      const offsetX = (containerWidth - renderedWidth) / 2;
      const offsetY = (containerHeight - renderedHeight) / 2;

      const toCanvas = (p: { x: number; y: number }) => ({
        x: offsetX + (isReferenceMirrored ? 1 - p.x : p.x) * renderedWidth,
        y: offsetY + p.y * renderedHeight,
      });

      const chunkFeedbackReplayFocus =
        chunkReviewPlaybackRef.current &&
        learnDrillRef.current?.step === "chunk_feedback" &&
        chunkDrillFocusJointRef.current;
      drawSkeleton(ctx, joints, toCanvas, "#00FF88", "#FFFFFF", {
        focusJoint: chunkFeedbackReplayFocus ? chunkDrillFocusJointRef.current : null,
        highlightColor: "#FBBF24",
      });

      const fj = chunkDrillFocusJointRef.current;
      if (chunkFeedbackReplayFocus && fj) {
        const def = getJointAngleDef(fj);
        if (def) {
          const [na, nb, nc] = def.points;
          const ja = joints[na];
          const jb = joints[nb];
          const jc = joints[nc];
          if (ja && jb && jc) {
            const angMap = calculateJointAngles(joints);
            const angVal = angMap[def.name];
            if (angVal !== undefined) {
              drawJointAngleArc(ctx, toCanvas(ja), toCanvas(jb), toCanvas(jc), angVal, "#FBBF24");
            }
          }
        }
      }

      rafId = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(rafId);
  }, [showSkeleton, skeletonFrames, isReferenceMirrored]);

  // Start practice mode: init webcam (or test video) and MoveNet
  const startPractice = useCallback(async (testMode = false) => {
    if (practiceMode) return;
    
    setWebcamLoading(true);
    setWebcamError(null);
    setUseTestVideo(testMode);
    setPositioningWarning(null);
    positioningOkRef.current = false;
    setSessionMirrorMapping("pending");

    try {
      if (!testMode) {
        // Request webcam access
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
          audio: false,
        });
        
        webcamStreamRef.current = stream;
      }

      // Load TensorFlow.js and MoveNet model
      const tf = await import("@tensorflow/tfjs");
      await tf.ready();
      // Load MoveNet SinglePose Lightning model from local files (avoids CORS issues)
      const model = await tf.loadGraphModel("/models/movenet/model.json");
      
      detectorRef.current = model;
      
      // Set practice mode first so the video element renders; show skeleton so feedback works
      setPracticeMode(true);
      setShowSkeleton(true);
      if (!testMode) {
        // Learn chunk drill opens body-gate at "Start your turn"; Dance opens immediately.
        setPositioningActive(practiceModeType === "dance");
        console.info(
          "%c[chunk-drill] Chunk-drill logs print in this browser tab: DevTools → Console (F12 or Cmd+Option+J). They do not appear in the terminal running npm run dev.",
          "color:#34d399;font-weight:bold;font-size:13px"
        );
      }
      
    } catch (err) {
      setWebcamError(err instanceof Error ? err.message : "Failed to access webcam");
      // Cleanup on error
      if (webcamStreamRef.current) {
        webcamStreamRef.current.getTracks().forEach((t) => t.stop());
        webcamStreamRef.current = null;
      }
    } finally {
      setWebcamLoading(false);
    }
  }, [practiceMode, practiceModeType]);
  
  // Effect to connect stream to video element once it's rendered (webcam mode only)
  useEffect(() => {
    if (!practiceMode || useTestVideo || !webcamRef.current || !webcamStreamRef.current) return;
    
    const video = webcamRef.current;
    const stream = webcamStreamRef.current;
    video.srcObject = stream;
    
    video.play()
      .then(() => {
        // detectorReady will be set after positioning preflight passes
      })
      .catch(() => {
        // ignore autoplay errors
      });
      
  }, [practiceMode, useTestVideo]);

  // In test mode, just mark detector as ready (we'll read from reference video directly)
  useEffect(() => {
    if (!practiceMode || !useTestVideo) return;
    setDetectorReady(true);
  }, [practiceMode, useTestVideo]);

  // If user selected a phrase before pressing Start Practice, start Learn chunk drill automatically now.
  useEffect(() => {
    if (!practiceMode || useTestVideo || practiceModeType !== "learn") return;
    if (learnDrill || activeSegmentId === null) return;
    if (completedLearnDrillSegmentId === activeSegmentId) return;
    const seg = segments.find((s) => s.segment_id === activeSegmentId);
    if (!seg) return;
    console.log("[chunk-drill] auto-starting selected phrase after Start Practice", {
      segmentId: activeSegmentId,
    });
    playSegment(seg);
  }, [
    practiceMode,
    useTestVideo,
    practiceModeType,
    learnDrill,
    activeSegmentId,
    completedLearnDrillSegmentId,
    segments,
    playSegment,
  ]);

  // Stop practice mode
  const stopPractice = useCallback(() => {
    if (webcamRafRef.current) {
      cancelAnimationFrame(webcamRafRef.current);
      webcamRafRef.current = null;
    }
    if (webcamStreamRef.current) {
      webcamStreamRef.current.getTracks().forEach((t) => t.stop());
      webcamStreamRef.current = null;
    }
    if (webcamRef.current) {
      webcamRef.current.srcObject = null;
      webcamRef.current.src = "";
    }
    detectorRef.current = null;
    jointAnglesBufferRef.current = []; // Clear the rolling buffer
    jointConfidenceBufferRef.current = [];
    setDetectorReady(false);
    setPracticeMode(false);
    setUseTestVideo(false);
    setPositioningActive(false);
    setPositioningWarning(null);
    positioningOkRef.current = false;
    setPoseScore(null);
    setDisplayedScore(null);
    setWorstJoint(null);
    setLearnDrill(null);
    learnDrillRef.current = null;
    learnCaptureRef.current = "none";
    learnAttemptBufferRef.current = [];
    learnAttemptConfidenceBufferRef.current = [];
    learnAttemptKeypointsBufferRef.current = [];
    prevAttemptSmoothedJointsRef.current = null;
    setAttemptReviewKeypoints([]);
    setChunkReviewPlayback(false);
    replayJointsRef.current = null;
    setChunkDrillFocusJoint(null);
    setReviewPlaybackTimeMs(0);
    dancePerformanceLogRef.current = [];
    setPerformanceReview(null);
    setTestDiagnostics(null);
    putTogetherStartedRef.current = false;
    queuedAttemptStartRef.current = false;
    setPreAttemptCountdownSec(null);
    setSessionMirrorMapping("pending");
    setCompletedLearnDrillSegmentId(null);
    setFullPhraseEvalResult(null);
    putTogetherKeyframesRef.current = [];
    putTogetherSmoothedRef.current = null;
    setPhraseReviewLearnKeyframes([]);
    dancePhraseKeyframesRef.current = [];
    dancePhraseSmoothedRef.current = null;
    setPhraseReviewDance(null);
    setPhrasePrepCountdownSec(null);
  }, []);

  const endDanceSession = useCallback(async () => {
    const entries = dancePerformanceLogRef.current;
    if (entries.length === 0) {
      setPerformanceReview({
        overall_assessment:
          useTestVideo
            ? "No test samples were logged yet. In Test Mode, play a phrase (or the full video) so we can measure self-compare alignment."
            : "No samples were logged yet. In Dance mode, play a phrase and move with the reference — we record joint differences over time.",
        top_phrases: [],
        next_session_focus: useTestVideo
          ? "Press play in Test Mode, let it run for at least a few seconds, then end session."
          : "Pick a phrase, press play, and dance along before ending the session.",
      });
      return;
    }

    // Test Mode should behave as a deterministic calibration pass, not dancer critique.
    if (useTestVideo) {
      const phraseStats = new Map<
        number,
        { samples: number; diffSum: number; worstJoint: string; worstDiff: number }
      >();
      let globalSum = 0;
      let globalCount = 0;
      let globalWorstJoint = "none";
      let globalWorstDiff = 0;

      for (const e of entries) {
        const values = Object.entries(e.per_joint_abs_diff ?? {});
        if (values.length === 0) continue;
        const mean = values.reduce((sum, [, v]) => sum + v, 0) / values.length;
        globalSum += mean;
        globalCount += 1;

        let localWorstJoint = "none";
        let localWorstDiff = 0;
        for (const [joint, diff] of values) {
          if (diff > localWorstDiff) {
            localWorstDiff = diff;
            localWorstJoint = joint;
          }
          if (diff > globalWorstDiff) {
            globalWorstDiff = diff;
            globalWorstJoint = joint;
          }
        }

        const curr = phraseStats.get(e.segment_id) ?? {
          samples: 0,
          diffSum: 0,
          worstJoint: localWorstJoint,
          worstDiff: localWorstDiff,
        };
        curr.samples += 1;
        curr.diffSum += mean;
        if (localWorstDiff > curr.worstDiff) {
          curr.worstDiff = localWorstDiff;
          curr.worstJoint = localWorstJoint;
        }
        phraseStats.set(e.segment_id, curr);
      }

      const overallMean = globalCount > 0 ? globalSum / globalCount : 0;
      const topPhrases = [...phraseStats.entries()]
        .map(([segment_id, s]) => ({
          segment_id,
          meanDiff: s.samples > 0 ? s.diffSum / s.samples : 0,
          worstJoint: s.worstJoint,
          worstDiff: s.worstDiff,
        }))
        .sort((a, b) => b.meanDiff - a.meanDiff)
        .slice(0, 3);

      const overall_assessment =
        overallMean <= 8
          ? `Test mode calibration looks healthy. Average self-compare error is ${overallMean.toFixed(1)} degrees, which is in the expected range.`
          : overallMean <= 15
            ? `Test mode calibration is decent but not perfect. Average self-compare error is ${overallMean.toFixed(1)} degrees; small pipeline/timing drift may still be present.`
            : `Test mode calibration is poor. Average self-compare error is ${overallMean.toFixed(1)} degrees (worst observed ${globalWorstDiff.toFixed(1)} on ${globalWorstJoint}), which suggests alignment/calibration issues rather than dancer technique.`;

      setPerformanceReview({
        overall_assessment,
        top_phrases: topPhrases.map((p) => ({
          segment_id: p.segment_id,
          reason: `Mean error ${p.meanDiff.toFixed(1)} degrees; worst ${p.worstJoint} at ${p.worstDiff.toFixed(1)} degrees.`,
        })),
        next_session_focus:
          overallMean <= 8
            ? "Calibration looks good; normal Dance mode feedback should be reasonably trustworthy."
            : "Tune test calibration first (timing/pose alignment) before trusting technique-focused Dance mode feedback.",
      });
      return;
    }

    setPerformanceReviewLoading(true);
    try {
      const res = await fetch(`${API_URL}/performance-review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ entries }),
      });
      if (!res.ok) throw new Error("Review failed");
      const data = (await res.json()) as {
        overall_assessment: string;
        top_phrases: { segment_id?: number; reason?: string }[];
        next_session_focus: string;
      };
      setPerformanceReview(data);
    } catch {
      setPerformanceReview({
        overall_assessment: "We could not generate a review right now.",
        top_phrases: [],
        next_session_focus: "Try again in a moment.",
      });
    } finally {
      setPerformanceReviewLoading(false);
    }
  }, [useTestVideo]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopPractice();
    };
  }, [stopPractice]);

  // Test mode: run pose detection on reference video (no webcam panel needed)
  useEffect(() => {
    if (!practiceMode || !detectorReady || !useTestVideo) return;

    let running = true;
    let isDetecting = false;

    const detector = detectorRef.current;
    const refVideo = videoRef.current;

    if (!detector || !refVideo) {
      return;
    }

    async function runPoseDetection() {
      if (!running || isDetecting) return;
      if (!refVideo || refVideo.videoWidth === 0 || refVideo.videoHeight === 0 || refVideo.paused) {
        // Skip detection if video not ready or paused
        if (running) setTimeout(runPoseDetection, 100);
        return;
      }
      
      isDetecting = true;
      
      try {
        const tf = await import("@tensorflow/tfjs");
        
        const inputTensor = tf.tidy(() => {
          const img = tf.browser.fromPixels(refVideo);
          const resized = tf.image.resizeBilinear(img, [192, 192]);
          const casted = tf.cast(resized, "int32");
          return tf.expandDims(casted, 0);
        });

        const result = (detector as { predict: (input: unknown) => { data: () => Promise<Float32Array>; dispose: () => void } }).predict(inputTensor);
        const data = await result.data();
        
        inputTensor.dispose();
        result.dispose();

        // Parse keypoints (no mirroring in test mode)
        const joints: Record<string, { x: number; y: number }> = {};
        const keypointScores: Record<string, number> = {};
        for (let i = 0; i < 17; i++) {
          const y = data[i * 3];
          const x = data[i * 3 + 1];
          const score = data[i * 3 + 2];
          
          if (score > 0.3) {
            const name = MOVENET_KEYPOINTS[i];
            joints[name] = { x, y };
            keypointScores[name] = score;
          }
        }
        
        latestUserJointsRef.current = joints;
        
        // Calculate angles and add to rolling buffer
        const angles = calculateJointAngles(joints);
        if (Object.keys(angles).length > 0) {
          jointAnglesBufferRef.current.push(angles);
          if (jointAnglesBufferRef.current.length > BUFFER_SIZE) {
            jointAnglesBufferRef.current.shift();
          }
        }

        const jointConf = calculateJointConfidence(keypointScores);
        if (Object.keys(jointConf).length > 0) {
          jointConfidenceBufferRef.current.push(jointConf);
          if (jointConfidenceBufferRef.current.length > BUFFER_SIZE) {
            jointConfidenceBufferRef.current.shift();
          }
        }
      } catch (err) {
        // ignore individual frame errors
      }
      
      isDetecting = false;
      
      if (running) {
        setTimeout(runPoseDetection, 33);
      }
    }

    runPoseDetection();

    return () => {
      running = false;
    };
  }, [practiceMode, detectorReady, useTestVideo]);

  // Webcam mode: draw webcam feed and user skeleton
  useEffect(() => {
    if (!practiceMode || !detectorReady || useTestVideo) return;

    let running = true;
    let frameCount = 0;
    let latestJoints: Record<string, { x: number; y: number }> | null = null;
    let isDetecting = false;

    // Small delay to ensure DOM has rendered the webcam panel
    const startTimeout = setTimeout(() => {
      const webcam = webcamRef.current;
      const canvas = webcamCanvasRef.current;
      const container = webcamContainerRef.current;
      const detector = detectorRef.current;

      if (!webcam || !canvas || !container || !detector) {
        return;
      }
      
      drawLoop();
      runPoseDetection();

      function drawLoop() {
        if (!running) return;

        frameCount++;

        if (!webcam || !canvas || !container) return;
        const rect = container.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        canvas.style.width = `${rect.width}px`;
        canvas.style.height = `${rect.height}px`;

        const ctx = canvas.getContext("2d");
        if (!ctx || webcam.videoWidth === 0 || webcam.videoHeight === 0) {
          webcamRafRef.current = requestAnimationFrame(drawLoop);
          return;
        }

        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.scale(dpr, dpr);

        const videoWidth = webcam.videoWidth;
        const videoHeight = webcam.videoHeight;
        const containerWidth = rect.width;
        const containerHeight = rect.height;
        
        const scale = containerWidth / videoWidth;
        const renderedWidth = containerWidth;
        const renderedHeight = videoHeight * scale;
        const offsetX = 0;
        const offsetY = Math.max(0, (containerHeight - renderedHeight) / 2);

        const reviewing = chunkReviewPlaybackRef.current;
        const skeletonJoints = reviewing ? replayJointsRef.current : latestJoints;

        if (!reviewing) {
          ctx.save();
          ctx.translate(containerWidth, offsetY);
          ctx.scale(-1, 1);
          ctx.drawImage(webcam, 0, 0, renderedWidth, renderedHeight);
          ctx.restore();
        } else {
          ctx.fillStyle = "#111111";
          ctx.fillRect(0, offsetY, containerWidth, renderedHeight);
        }

        if (skeletonJoints && Object.keys(skeletonJoints).length > 0) {
          const toCanvas = (p: { x: number; y: number }) => ({
            x: offsetX + (reviewing && isReferenceMirrored ? 1 - p.x : p.x) * renderedWidth,
            y: offsetY + p.y * renderedHeight,
          });

          const chunkFeedbackReplayFocus =
            reviewing &&
            learnDrillRef.current?.step === "chunk_feedback" &&
            chunkDrillFocusJointRef.current;
          drawSkeleton(ctx, skeletonJoints, toCanvas, "#FF6B6B", "#FFFFFF", {
            focusJoint: chunkFeedbackReplayFocus ? chunkDrillFocusJointRef.current : null,
            highlightColor: "#FBBF24",
            baseLineWidth: 4,
            jointRadius: 6,
          });

          const fj = chunkDrillFocusJointRef.current;
          if (chunkFeedbackReplayFocus && fj) {
            const def = getJointAngleDef(fj);
            if (def) {
              const [na, nb, nc] = def.points;
              const ja = skeletonJoints[na];
              const jb = skeletonJoints[nb];
              const jc = skeletonJoints[nc];
              if (ja && jb && jc) {
                const angMap = calculateJointAngles(skeletonJoints);
                const angVal = angMap[def.name];
                if (angVal !== undefined) {
                  drawJointAngleArc(ctx, toCanvas(ja), toCanvas(jb), toCanvas(jc), angVal, "#FBBF24");
                }
              }
            }
          }
        }

        webcamRafRef.current = requestAnimationFrame(drawLoop);
      }

      async function runPoseDetection() {
        if (!running || isDetecting) return;
        if (!webcam) return;
        if (chunkReviewPlaybackRef.current) {
          isDetecting = false;
          if (running) setTimeout(runPoseDetection, 33);
          return;
        }

        isDetecting = true;
        
        try {
          const tf = await import("@tensorflow/tfjs");
          
          const inputTensor = tf.tidy(() => {
            const img = tf.browser.fromPixels(webcam);
            const resized = tf.image.resizeBilinear(img, [192, 192]);
            const casted = tf.cast(resized, "int32");
            return tf.expandDims(casted, 0);
          });

          const result = (detector as { predict: (input: unknown) => { data: () => Promise<Float32Array>; dispose: () => void } }).predict(inputTensor);
          const data = await result.data();
          
          inputTensor.dispose();
          result.dispose();

          const joints: Record<string, { x: number; y: number }> = {};
          const keypointScores: Record<string, number> = {};
          for (let i = 0; i < 17; i++) {
            const y = data[i * 3];
            const x = data[i * 3 + 1];
            const score = data[i * 3 + 2];
            
            if (score > 0.3) {
              const name = MOVENET_KEYPOINTS[i];
              joints[name] = {
                x: 1 - x, // Mirror for selfie view
                y: y,
              };
              keypointScores[name] = score;
            }
          }
          
          latestJoints = joints;
          latestUserJointsRef.current = joints;

          if (learnCaptureRef.current === "chunk_attempt") {
            const smoothedJoints = smoothJointsEma(joints, prevAttemptSmoothedJointsRef.current, 0.3);
            prevAttemptSmoothedJointsRef.current = smoothedJoints;
            learnAttemptKeypointsBufferRef.current.push({
              timestamp_ms: (videoRef.current?.currentTime ?? 0) * 1000,
              joints: { ...smoothedJoints },
            });
          }

          if (
            practiceModeType === "dance" &&
            learnCaptureRef.current === "none" &&
            !learnDrillRef.current &&
            !phraseReviewDanceRef.current &&
            skeletonFrames.length > 0
          ) {
            const tMs = (videoRef.current?.currentTime ?? 0) * 1000;
            const seg =
              activeSegmentId != null
                ? segments.find(
                    (s) =>
                      s.segment_id === activeSegmentId &&
                      tMs >= s.start_time_ms &&
                      tMs <= s.end_time_ms
                  )
                : segments.find((s) => tMs >= s.start_time_ms && tMs <= s.end_time_ms);
            if (seg) {
              const arr = dancePhraseKeyframesRef.current;
              const lastTs = arr.length ? arr[arr.length - 1]?.timestamp_ms : -Infinity;
              if (tMs - lastTs >= 76) {
                const smoothedDance = smoothJointsEma(joints, dancePhraseSmoothedRef.current, 0.3);
                dancePhraseSmoothedRef.current = smoothedDance;
                dancePhraseKeyframesRef.current.push({
                  timestamp_ms: tMs,
                  joints: { ...smoothedDance },
                });
              }
            }
          }

          // Calculate angles — learn chunk / put-together use dedicated buffers
          const angles = calculateJointAngles(joints);
          if (Object.keys(angles).length > 0) {
            if (learnCaptureRef.current === "chunk_attempt") {
              learnAttemptBufferRef.current.push(angles);
            } else if (learnCaptureRef.current === "put_together") {
              putTogetherBufferRef.current.push(angles);
              const smoothedPut = smoothJointsEma(joints, putTogetherSmoothedRef.current, 0.3);
              putTogetherSmoothedRef.current = smoothedPut;
              const tMs = (videoRef.current?.currentTime ?? 0) * 1000;
              const pk = putTogetherKeyframesRef.current;
              if (pk.length === 0 || tMs - (pk[pk.length - 1]?.timestamp_ms ?? 0) >= 72) {
                putTogetherKeyframesRef.current.push({ timestamp_ms: tMs, joints: { ...smoothedPut } });
              }
            } else {
              jointAnglesBufferRef.current.push(angles);
              if (jointAnglesBufferRef.current.length > BUFFER_SIZE) {
                jointAnglesBufferRef.current.shift();
              }
            }
          }

          const jointConf = calculateJointConfidence(keypointScores);
          if (Object.keys(jointConf).length > 0) {
            if (learnCaptureRef.current === "chunk_attempt") {
              learnAttemptConfidenceBufferRef.current.push(jointConf);
            } else if (learnCaptureRef.current === "none") {
              jointConfidenceBufferRef.current.push(jointConf);
              if (jointConfidenceBufferRef.current.length > BUFFER_SIZE) {
                jointConfidenceBufferRef.current.shift();
              }
            }
          }

          // If we lose ankles for a while during practice, re-activate positioning gate
          const leftAnkleScore = data[15 * 3 + 2];
          const rightAnkleScore = data[16 * 3 + 2];
          if (leftAnkleScore > 0.3 && rightAnkleScore > 0.3) {
            lostBodyCountRef.current = 0;
          } else {
            lostBodyCountRef.current += 1;
          }

          // After ~0.5s of missing ankles, re-gate — but not while user is idle in chunk drill
          // (preview / chunk_feedback) or during the pre-attempt countdown, so stepping
          // back to tap "Start your turn" doesn't flip detectorReady off.
          const ldGate = learnDrillRef.current;
          const suppressLostBodyGate =
            chunkReviewPlaybackRef.current ||
            (ldGate &&
              (ldGate.step === "preview" ||
                ldGate.step === "chunk_feedback" ||
                ldGate.step === "phrase_prep" ||
                ldGate.step === "phrase_review"));
          if (
            !suppressLostBodyGate &&
            !positioningActive &&
            detectorReady &&
            lostBodyCountRef.current >= 15
          ) {
            positioningOkRef.current = false;
            setDetectorReady(false);
            setPositioningActive(true);
            setPositioningWarning("Move further back — we can't see your full body.");
            lostBodyCountRef.current = 0;
          }
        } catch (err) {
          // ignore individual frame errors
        }
        
        isDetecting = false;
        
        if (running) {
          setTimeout(runPoseDetection, 33);
        }
      }
    }, 100);

    return () => {
      running = false;
      clearTimeout(startTimeout);
      if (webcamRafRef.current) {
        cancelAnimationFrame(webcamRafRef.current);
      }
    };
  }, [
    practiceMode,
    detectorReady,
    useTestVideo,
    isReferenceMirrored,
    practiceModeType,
    activeSegmentId,
    segments,
    skeletonFrames.length,
  ]);

  // Positioning preflight: require ankles visible before starting session
  useEffect(() => {
    if (!practiceMode || useTestVideo || !positioningActive) return;
    const webcam = webcamRef.current;
    const detector = detectorRef.current;
    if (!webcam || !detector) return;

    let running = true;
    let okStreak = 0;
    const start = Date.now();

    const run = async () => {
      if (!running) return;
      if (positioningOkRef.current) return;

      if (webcam.videoWidth === 0 || webcam.videoHeight === 0 || webcam.readyState < 2) {
        setTimeout(run, 100);
        return;
      }

      try {
        const tf = await import("@tensorflow/tfjs");
        const inputTensor = tf.tidy(() => {
          const img = tf.browser.fromPixels(webcam);
          const resized = tf.image.resizeBilinear(img, [192, 192]);
          const casted = tf.cast(resized, "int32");
          return tf.expandDims(casted, 0);
        });

        const result = (detector as { predict: (input: unknown) => { data: () => Promise<Float32Array>; dispose: () => void } }).predict(inputTensor);
        const data = await result.data();
        inputTensor.dispose();
        result.dispose();

        // MoveNet indices: LEFT_ANKLE=15, RIGHT_ANKLE=16
        const leftAnkleScore = data[15 * 3 + 2];
        const rightAnkleScore = data[16 * 3 + 2];

        if (leftAnkleScore > 0.3 && rightAnkleScore > 0.3) {
          okStreak += 1;
        } else {
          okStreak = 0;
        }

        const elapsed = Date.now() - start;
        if (elapsed > 3000 && okStreak === 0) {
          setPositioningWarning("Move further back — we can't see your full body.");
        }

        // Require a few consecutive frames to avoid flicker
        if (okStreak >= 6) {
          positioningOkRef.current = true;
          setPositioningActive(false);
          setPositioningWarning(null);
          setDetectorReady(true);
          return;
        }
      } catch {
        // ignore
      }

      setTimeout(run, 100);
    };

    run();

    return () => {
      running = false;
    };
  }, [practiceMode, useTestVideo, positioningActive]);

  // Send coaching requests to backend (throttled or chunk-eval)
  const sendCoachingRequest = useCallback(
    async (
      segmentId: number,
      userMedianAngles: Record<string, number>,
      userMedianConfidence: Record<string, number>,
      validJoints: string[],
      matchLevel: "good" | "developing" | "needs_work",
      mode: "learn" | "dance",
      opts?: {
        focus_joint?: string;
        diff_threshold_degrees?: number;
        reference_angle_summary_override?: Record<string, number>;
      }
    ) => {
      if (!jobId) return;

      const segment = segments.find((s) => s.segment_id === segmentId);
      const reference_angle_summary =
        opts?.reference_angle_summary_override ?? segment?.angle_summary ?? {};

      try {
        setCoachingLoading(true);

        const body: Record<string, unknown> = {
          segment_id: segmentId,
          reference_angle_summary,
          user_angles: userMedianAngles,
          user_joint_confidence: userMedianConfidence,
          valid_joints: validJoints,
          match_level: matchLevel,
          skill_level: "beginner",
          style: "contemporary",
          practice_mode: mode,
        };
        if (opts?.focus_joint) body.focus_joint = opts.focus_joint;
        if (opts?.diff_threshold_degrees != null) body.diff_threshold_degrees = opts.diff_threshold_degrees;

        const res = await fetch(`${API_URL}/coaching`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(body),
        });

        if (!res.ok) {
          return;
        }

        const data = (await res.json()) as { note?: string };
        if (data.note) {
          if (learnDrillRef.current?.step !== "chunk_feedback") {
            return;
          }
          setCoachingNote(data.note);
        }
      } catch (err) {
        // swallow coaching errors; UI should keep working
      } finally {
        setCoachingLoading(false);
      }
    },
    [jobId, segments]
  );

  const beginChunkAttempt = useCallback((ld: LearnDrillState) => {
    const video = videoRef.current;
    if (!video) {
      setPreAttemptCountdownSec(null);
      return;
    }
    setPreAttemptCountdownSec(null);
    const c = ld.chunks[ld.chunkIndex];
    learnAttemptBufferRef.current = [];
    learnAttemptConfidenceBufferRef.current = [];
    learnAttemptKeypointsBufferRef.current = [];
    prevAttemptSmoothedJointsRef.current = null;
    setChunkReviewPlayback(false);
    replayJointsRef.current = null;
    setChunkDrillFocusJoint(null);
    learnCaptureRef.current = "chunk_attempt";
    if (loopIntervalRef.current) {
      clearInterval(loopIntervalRef.current);
      loopIntervalRef.current = null;
    }
    const fromChunkFeedback = ld.step === "chunk_feedback";
    const next = { ...ld, step: "attempt" as const };
    learnDrillRef.current = next;
    setLearnDrill(next);
    video.currentTime = c.startMs / 1000;
    video.playbackRate = 0.5;
    video.play();
    console.log(
      `[chunk-drill] STATE ${fromChunkFeedback ? "chunk_feedback" : "preview"} → attempt`,
      {
        chunkIndex: ld.chunkIndex,
        startMs: c.startMs,
        endMs: c.endMs,
      }
    );
    console.log("[chunk-drill] attempt recording START (buffer cleared, MoveNet frames = 0)");
  }, []);

  const startChunkAttemptFromPreview = useCallback(() => {
    const ld = learnDrillRef.current;
    if (!ld || ld.step !== "preview") return;
    if (preAttemptCountdownRef.current !== null) return;
    if (!detectorReady) {
      queuedAttemptStartRef.current = true;
      setPositioningWarning("Move further back until both ankles are visible, then we'll begin your turn automatically.");
      setPositioningActive(true);
      console.log("[chunk-drill] Start your turn clicked — waiting for body gate to pass, then auto-starting attempt");
      return;
    }
    setPreAttemptCountdownSec(3);
    console.log("[chunk-drill] pre-attempt countdown started (3…2…1)");
  }, [detectorReady]);

  const retryChunkFromFeedback = useCallback(() => {
    const ld = learnDrillRef.current;
    if (!ld || ld.step !== "chunk_feedback") return;
    const n = ld.chunkAttemptsOnSlice ?? 0;
    if (n >= 3) return;
    if (preAttemptCountdownRef.current !== null) return;
    setChunkReviewPlayback(false);
    replayJointsRef.current = null;
    setCoachingNote(null);
    if (!detectorReady) {
      queuedAttemptStartRef.current = true;
      setPositioningWarning("Move further back until both ankles are visible, then we'll auto-start retry.");
      setPositioningActive(true);
      console.log("[chunk-drill] Retry clicked — waiting for body gate to pass, then auto-starting attempt");
      return;
    }
    setPreAttemptCountdownSec(3);
    console.log("[chunk-drill] pre-attempt countdown started (retry, 3…2…1)");
  }, [detectorReady]);

  // If a chunk attempt was queued while the body gate was visible, start countdown once gate passes.
  useEffect(() => {
    if (!queuedAttemptStartRef.current) return;
    if (!practiceMode || useTestVideo || positioningActive || !detectorReady) return;
    const ld = learnDrillRef.current;
    if (!ld || (ld.step !== "preview" && ld.step !== "chunk_feedback")) return;
    queuedAttemptStartRef.current = false;
    console.log("[chunk-drill] body gate passed — starting pre-attempt countdown");
    setCoachingNote(null);
    setPreAttemptCountdownSec(3);
  }, [practiceMode, useTestVideo, positioningActive, detectorReady]);

  // 3 → 2 → 1 → 0, then begin chunk attempt (recording + video).
  useEffect(() => {
    if (preAttemptCountdownSec === null) return;
    if (preAttemptCountdownSec === 0) {
      const ld = learnDrillRef.current;
      if (ld && (ld.step === "preview" || ld.step === "chunk_feedback")) {
        beginChunkAttempt(ld);
      }
      setPreAttemptCountdownSec(null);
      return;
    }
    const t = window.setTimeout(() => {
      setPreAttemptCountdownSec((c) => (c === null ? null : c - 1));
    }, 1000);
    return () => clearTimeout(t);
  }, [preAttemptCountdownSec, beginChunkAttempt]);

  // Entering full phrase: start countdown + spoken cue after last-chunk feedback.
  useEffect(() => {
    if (learnDrill?.step !== "phrase_prep") return;
    setPhrasePrepCountdownSec(PHRASE_PREP_COUNTDOWN_SEC);
    if (typeof window !== "undefined" && typeof window.speechSynthesis !== "undefined") {
      const synth = window.speechSynthesis;
      synth.cancel();
      const utterance = new SpeechSynthesisUtterance("Now put it all together!");
      utterance.rate = 0.98;
      const voice = ttsVoiceRef.current;
      if (voice) utterance.voice = voice;
      synth.speak(utterance);
    }
  }, [learnDrill?.step, learnDrill?.segmentId]);

  // 3 → 2 → 1 → 0, then start put_together capture.
  useEffect(() => {
    if (phrasePrepCountdownSec === null) return;
    if (phrasePrepCountdownSec === 0) {
      setPhrasePrepCountdownSec(null);
      setLearnDrill((prev) =>
        prev?.step === "phrase_prep" ? { ...prev, step: "put_together" } : prev
      );
      return;
    }
    const t = window.setTimeout(() => {
      setPhrasePrepCountdownSec((c) => (c === null ? null : c - 1));
    }, 1000);
    return () => clearTimeout(t);
  }, [phrasePrepCountdownSec]);

  // Attempt: countdown + transition to chunk feedback after recording
  useEffect(() => {
    const v = videoRef.current;
    if (!learnDrill || learnDrill.step !== "attempt" || !v) return;
    const c = learnDrill.chunks[learnDrill.chunkIndex];
    if (!c) return;
    const onTimeUpdate = () => {
      const remain = Math.max(0, Math.ceil(c.endMs / 1000 - v.currentTime));
      setAttemptCountdownSec(remain);
      if (v.currentTime >= c.endMs / 1000 - 0.12) {
        // `timeupdate` can fire repeatedly near the cut — only finalize once, and never snapshot after
        // the buffer was cleared (second fire would call setAttemptReviewKeypoints([]) and wipe replay).
        if (learnCaptureRef.current !== "chunk_attempt") return;
        v.removeEventListener("timeupdate", onTimeUpdate);
        v.pause();
        const frameCount = learnAttemptBufferRef.current.length;
        const keyframes = learnAttemptKeypointsBufferRef.current.map((frame) => ({
          timestamp_ms: frame.timestamp_ms,
          joints: { ...frame.joints },
        }));
        learnAttemptKeypointsBufferRef.current = [];
        setAttemptReviewKeypoints(keyframes);
        learnCaptureRef.current = "none";
        console.log("[chunk-drill] attempt recording END", {
          chunkIndex: learnDrill.chunkIndex,
          movenetFramesInBuffer: frameCount,
          keyframeFramesInBuffer: keyframes.length,
        });
        console.log("[chunk-drill] STATE attempt → chunk_feedback", { chunkIndex: learnDrill.chunkIndex });
        setLearnDrill((prev) =>
          prev
            ? {
                ...prev,
                step: "chunk_feedback",
                chunkAttemptsOnSlice: (prev.chunkAttemptsOnSlice ?? 0) + 1,
              }
            : prev
        );
      }
    };
    v.addEventListener("timeupdate", onTimeUpdate);
    onTimeUpdate();
    return () => v.removeEventListener("timeupdate", onTimeUpdate);
  }, [learnDrill?.step, learnDrill?.chunkIndex]);

  // Chunk drill feedback (no score gate): highlight worst joint + coaching note; user chooses retry vs continue.
  useEffect(() => {
    const ld = learnDrillRef.current;
    if (!ld || ld.step !== "chunk_feedback") return;
    if (!skeletonFrames.length) {
      console.log("[chunk-drill] feedback SKIPPED (no skeleton frames loaded yet)");
      return;
    }
    const attemptNum = ld.chunkAttemptsOnSlice ?? 0;
    const dedupeKey = `${ld.chunkIndex}-${attemptNum}`;
    if (chunkFeedbackProcessedKeyRef.current === dedupeKey) return;
    chunkFeedbackProcessedKeyRef.current = dedupeKey;

    const chunk = ld.chunks[ld.chunkIndex];
    const attemptKfs = attemptReviewKeypointsRef.current;

    const chunkFrames = getSortedChunkSkeletonFrames(skeletonFrames, chunk.startMs, chunk.endMs);
    const {
      indices: kpIndices,
      velocities: kpVelocities,
      source: kpSource,
    } = selectKeyPoseFrameIndices(
      chunkFrames,
      KEY_POSE_MIN_SEPARATION_MS,
      KEY_POSE_COUNT_MIN,
      KEY_POSE_COUNT_MAX
    );
    const keyPoseTimes = kpIndices.map((i) => chunkFrames[i]?.timestamp_ms ?? 0);
    const keyPoseMeta = kpIndices.map((i) => ({
      skeletonWindowFrameIndex: i,
      refTimestampMs: chunkFrames[i]?.timestamp_ms ?? null,
      movementVelocity: kpVelocities[i] ?? null,
    }));

    const keyPoseCandidatesOk =
      kpIndices.length >= KEY_POSE_COUNT_MIN &&
      attemptKfs.length > 0 &&
      chunkFrames.length >= 2;

    const confBufForKeyPose = learnAttemptConfidenceBufferRef.current;
    const keyPosePickRows: { referenceTargetMs: number; pick: KeyPosePickResult }[] = [];

    if (keyPoseCandidatesOk) {
      for (const t of keyPoseTimes) {
        const pick = pickUserKeyframeForReferenceKeyPose(attemptKfs, confBufForKeyPose, t);
        if (pick) {
          keyPosePickRows.push({ referenceTargetMs: t, pick });
          console.log("[chunk-drill] key pose user keyframe selection", {
            referenceTargetMs: t,
            chosenKeyframeIndex: pick.keyframeIndex,
            chosenKeyframeMs: attemptKfs[pick.keyframeIndex]?.timestamp_ms,
            deltaMsFromTarget: Math.abs((attemptKfs[pick.keyframeIndex]?.timestamp_ms ?? 0) - t),
            candidateKeyframesTried: pick.attempts.length,
            attempts: pick.attempts,
          });
        } else {
          console.log("[chunk-drill] key pose dropped — no user keyframe met joint-count threshold after filters", {
            referenceTargetMs: t,
            angleRangePlausibleDeg: [
              KEY_POSE_USER_ANGLE_MIN_PLAUSIBLE,
              KEY_POSE_USER_ANGLE_MAX_PLAUSIBLE,
            ],
            perJointConfMin: KEY_POSE_PER_JOINT_CONF_MIN,
            minValidJointsRequired: KEY_POSE_MIN_VALID_USER_JOINTS,
          });
        }
      }
    }

    const useKeyPoseEval =
      keyPoseCandidatesOk && keyPosePickRows.length >= KEY_POSE_COUNT_MIN;

    console.log("[chunk-drill] evaluation RUN", {
      chunkIndex: ld.chunkIndex,
      windowMs: { start: chunk.startMs, end: chunk.endMs },
      bufferFrameCount: learnAttemptBufferRef.current.length,
      attemptKeyframeCount: attemptKfs.length,
      keyPoseEval: useKeyPoseEval,
      keyPosePicksAfterTrackingFilters: keyPosePickRows.length,
      keyPoseDetection: {
        skeletonFramesInChunk: chunkFrames.length,
        selectionSource: kpSource,
        selectedKeyPoseCount: kpIndices.length,
        keyPoses: keyPoseMeta,
      },
    });

    let userMed: Record<string, number>;
    let userMedConf: Record<string, number>;
    let refMedDirect: Record<string, number>;
    let refMedSwapped: Record<string, number>;
    let directAvg: { avgDiff: number; jointsCompared: number };
    let swappedAvg: { avgDiff: number; jointsCompared: number };
    let refMed: Record<string, number>;
    let evaluationMode: "key_pose" | "median_window";
    let highlight: { worstKey: string; worstLabel: string; worstDiff: number } | null = null;

    if (useKeyPoseEval) {
      evaluationMode = "key_pose";
      console.log("[chunk-drill] key pose timestamps (reference, movement valleys)", {
        refTimestampsMs: keyPosePickRows.map((r) => r.referenceTargetMs),
        minSeparationMs: KEY_POSE_MIN_SEPARATION_MS,
      });

      const pairsDirect = keyPosePickRows.map(({ referenceTargetMs: t, pick }) => ({
        user: filterUserAnglesForKeyPoseMirrorCompare(pick.user, pick.conf),
        ref: interpolatedAnglesAtTime(skeletonFrames, t),
      }));
      const pairsSwapped = keyPosePickRows.map(({ referenceTargetMs: t, pick }) => ({
        user: filterUserAnglesForKeyPoseMirrorCompare(pick.user, pick.conf),
        ref: swapLeftRightJointAngles(interpolatedAnglesAtTime(skeletonFrames, t)),
      }));

      directAvg = averageJointDiffAcrossKeyPoses(pairsDirect);
      swappedAvg = averageJointDiffAcrossKeyPoses(pairsSwapped);

      refMedDirect = meanJointAnglesAcrossSamples(
        keyPosePickRows.map(({ referenceTargetMs: t }) => interpolatedAnglesAtTime(skeletonFrames, t))
      );
      refMedSwapped = meanJointAnglesAcrossSamples(
        keyPosePickRows.map(({ referenceTargetMs: t }) =>
          swapLeftRightJointAngles(interpolatedAnglesAtTime(skeletonFrames, t))
        )
      );

      if (sessionMirrorMappingRef.current === "pending") {
        const lockSwapped = swappedAvg.avgDiff < directAvg.avgDiff;
        const locked: SessionMirrorMapping = lockSwapped ? "lr_swapped" : "direct";
        setSessionMirrorMapping(locked);
        console.log(`[mirror] mirror locked: ${lockSwapped ? "swapped" : "direct"} (key-pose avg diff)`);
      }
      const useSwappedMapping =
        sessionMirrorMappingRef.current === "pending"
          ? swappedAvg.avgDiff < directAvg.avgDiff
          : sessionMirrorMappingRef.current === "lr_swapped";

      const poses = keyPosePickRows.map(({ referenceTargetMs: t, pick }) => {
        const baseRef = interpolatedAnglesAtTime(skeletonFrames, t);
        const ref = useSwappedMapping ? swapLeftRightJointAngles(baseRef) : baseRef;
        return { user: pick.user, ref, conf: pick.conf };
      });

      userMed = meanJointAnglesAcrossSamples(
        keyPosePickRows.map(({ pick }) => filterUserAnglesForKeyPoseMirrorCompare(pick.user, pick.conf))
      );
      userMedConf = getMedianConfidence(poses.map((p) => p.conf));
      refMed = meanJointAnglesAcrossSamples(poses.map((p) => p.ref));
      highlight =
        pickHighlightJointFromKeyPosePoses(poses) ??
        pickHighlightJointForChunkFeedback(userMed, userMedConf, refMed);

      console.log("[chunk-drill] mirror avg diff check (key poses)", {
        directAvgDiff: Number.isFinite(directAvg.avgDiff) ? Number(directAvg.avgDiff.toFixed(2)) : null,
        swappedAvgDiff: Number.isFinite(swappedAvg.avgDiff) ? Number(swappedAvg.avgDiff.toFixed(2)) : null,
        directJointsCompared: directAvg.jointsCompared,
        swappedJointsCompared: swappedAvg.jointsCompared,
      });
      console.log("[chunk-drill] evaluation user mean joint angles (key poses)", userMed);
      console.log("[chunk-drill] evaluation reference mean joint angles (key poses, chosen mapping)", refMed);
      console.log(
        "[chunk-drill] evaluation reference mapping chosen",
        useSwappedMapping ? "lr_swapped" : "direct"
      );

      const perPoseLog = keyPosePickRows.map(({ referenceTargetMs: t, pick }, i) => {
        const ut = attemptKfs[pick.keyframeIndex]?.timestamp_ms ?? null;
        return {
          referenceTargetMs: t,
          closestUserKeyframeMs: ut,
          absDeltaMs: ut !== null ? Math.abs(t - ut) : null,
          pickAttempts: pick.attempts,
          diffsDeg: computeChunkJointDiffsForLog(poses[i]!.user, poses[i]!.ref),
        };
      });
      console.log("[chunk-drill] key pose per-target alignment & joint diffs (°)", perPoseLog);
    } else {
      evaluationMode = "median_window";
      console.log("[chunk-drill] key pose eval unavailable — falling back to median entire window", {
        reason:
          kpIndices.length < KEY_POSE_COUNT_MIN
            ? "not_enough_key_poses"
            : attemptKfs.length === 0
              ? "no_attempt_keyframes"
              : chunkFrames.length < 2
                ? "chunk_too_short"
                : keyPosePickRows.length < KEY_POSE_COUNT_MIN
                  ? "too_few_valid_user_keyframes_after_tracking_filters"
                  : "unknown",
        keyPosePicksAfterFilters: keyPosePickRows.length,
      });

      userMed = getMedianAngles(learnAttemptBufferRef.current);
      userMedConf = getMedianConfidence(learnAttemptConfidenceBufferRef.current);
      refMedDirect = medianAnglesForSkeletonWindow(skeletonFrames, chunk.startMs, chunk.endMs);
      refMedSwapped = swapLeftRightJointAngles(refMedDirect);
      directAvg = averageJointDiff(userMed, refMedDirect);
      swappedAvg = averageJointDiff(userMed, refMedSwapped);

      if (sessionMirrorMappingRef.current === "pending") {
        const lockSwapped = swappedAvg.avgDiff < directAvg.avgDiff;
        const locked: SessionMirrorMapping = lockSwapped ? "lr_swapped" : "direct";
        setSessionMirrorMapping(locked);
        console.log(`[mirror] mirror locked: ${lockSwapped ? "swapped" : "direct"}`);
      }
      const useSwappedMapping =
        sessionMirrorMappingRef.current === "pending"
          ? swappedAvg.avgDiff < directAvg.avgDiff
          : sessionMirrorMappingRef.current === "lr_swapped";
      refMed = useSwappedMapping ? refMedSwapped : refMedDirect;
      highlight = pickHighlightJointForChunkFeedback(userMed, userMedConf, refMed);

      console.log("[chunk-drill] evaluation user median joint angles (from buffer)", userMed);
      console.log("[chunk-drill] evaluation user median joint confidence (from buffer)", userMedConf);
      console.log("[chunk-drill] mirror avg diff check", {
        directAvgDiff: Number.isFinite(directAvg.avgDiff) ? Number(directAvg.avgDiff.toFixed(2)) : null,
        swappedAvgDiff: Number.isFinite(swappedAvg.avgDiff) ? Number(swappedAvg.avgDiff.toFixed(2)) : null,
        directJointsCompared: directAvg.jointsCompared,
        swappedJointsCompared: swappedAvg.jointsCompared,
      });
      console.log("[chunk-drill] evaluation reference median joint angles (skeleton window)", refMedDirect);
      console.log(
        "[chunk-drill] evaluation reference mapping chosen",
        useSwappedMapping ? "lr_swapped" : "direct"
      );
    }

    console.log("[chunk-drill] evaluation mode", evaluationMode);

    if (Object.keys(userMed).length < 2) {
      console.log("[chunk-drill] Claude: SKIPPED (insufficient user joint coverage)", {
        jointCount: Object.keys(userMed).length,
      });
      setChunkDrillFocusJoint(null);
      setCoachingNote("We couldn't capture enough pose data in that window — try again or move on.");
      videoRef.current?.pause();
      return;
    }

    const diffsForLog = computeChunkJointDiffsForLog(userMed, refMed);
    console.log("[chunk-drill] joint diffs vs reference summary (°)", diffsForLog);
    console.log("[chunk-drill] highlight joint (largest diff, coaching focus)", {
      worstJoint: highlight?.worstKey ?? null,
      worstAbsDiffDeg: highlight?.worstDiff != null ? Number(highlight.worstDiff.toFixed(2)) : null,
    });

    setChunkDrillFocusJoint(highlight?.worstKey ?? null);
    const validJoints = Object.keys(userMed).filter((k) => userMed[k] !== 0);
    const fakeConf: Record<string, number> = {};
    for (const k of validJoints) fakeConf[k] = 0.5;
    const focusKey = highlight?.worstKey ?? validJoints[0] ?? JOINT_ANGLES[0]!.name;
    console.log("[chunk-drill] Claude: CALLING /coaching (chunk feedback)", {
      segmentId: ld.segmentId,
      focus_joint: focusKey,
    });
    void sendCoachingRequest(
      ld.segmentId,
      userMed,
      fakeConf,
      validJoints.length ? validJoints : [focusKey],
      "developing",
      "learn",
      {
        focus_joint: focusKey,
        reference_angle_summary_override: refMed,
      }
    );
    videoRef.current?.pause();
  }, [
    learnDrill?.step,
    learnDrill?.chunkIndex,
    learnDrill?.chunkAttemptsOnSlice,
    skeletonFrames.length,
    sendCoachingRequest,
  ]);

  const moveToNextChunkFromFeedback = useCallback(() => {
    const cur = learnDrillRef.current;
    if (!cur || cur.step !== "chunk_feedback") return;
    chunkFeedbackProcessedKeyRef.current = "";
    setLearnDrill((prev) => {
      if (!prev || prev.step !== "chunk_feedback") return prev;
      if (prev.chunkIndex + 1 >= prev.chunks.length) {
        console.log("[chunk-drill] STATE chunk_feedback → phrase_prep (countdown before full phrase)");
        return { ...prev, step: "phrase_prep", chunkAttemptsOnSlice: 0 };
      }
      console.log("[chunk-drill] STATE chunk_feedback → preview", {
        nextChunkIndex: prev.chunkIndex + 1,
      });
      return {
        ...prev,
        chunkIndex: prev.chunkIndex + 1,
        step: "preview",
        chunkAttemptsOnSlice: 0,
      };
    });
    setCoachingNote(null);
    setChunkDrillFocusJoint(null);
    setChunkReviewPlayback(false);
    replayJointsRef.current = null;
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
  }, []);

  /** After 3 attempts on a sub-chunk, always advance (same as explicitly tapping Move on). */
  useEffect(() => {
    if (!learnDrill || learnDrill.step !== "chunk_feedback") return;
    const n = learnDrill.chunkAttemptsOnSlice ?? 0;
    if (n < 3) return;
    const t = window.setTimeout(() => moveToNextChunkFromFeedback(), 1600);
    return () => clearTimeout(t);
  }, [learnDrill?.step, learnDrill?.chunkIndex, learnDrill?.chunkAttemptsOnSlice, moveToNextChunkFromFeedback]);

  // Put it together: full phrase once at 1x, then stop (stay on same phrase selection)
  useEffect(() => {
    if (!learnDrill || learnDrill.step !== "put_together") {
      putTogetherStartedRef.current = false;
      return;
    }
    if (putTogetherStartedRef.current) return;
    putTogetherStartedRef.current = true;
    const video = videoRef.current;
    if (!video) return;
    const seg = learnDrill.segment;
    console.log("[chunk-drill] STATE → put_together", {
      segmentId: seg.segment_id,
      phraseMs: { start: seg.start_time_ms, end: seg.end_time_ms },
    });
    putTogetherBufferRef.current = [];
    putTogetherKeyframesRef.current = [];
    putTogetherSmoothedRef.current = null;
    learnCaptureRef.current = "put_together";
    setPlaybackRate(1);
    video.playbackRate = 1;
    video.currentTime = seg.start_time_ms / 1000;
    if (loopIntervalRef.current) {
      clearInterval(loopIntervalRef.current);
      loopIntervalRef.current = null;
    }
    video.play();
    const onTimeUpdate = () => {
      if (video.currentTime >= seg.end_time_ms / 1000 - 0.12) {
        video.pause();
        video.removeEventListener("timeupdate", onTimeUpdate);
        learnCaptureRef.current = "none";
        setLearnDrill((prev) => (prev ? { ...prev, step: "phrase_review" } : prev));
        putTogetherStartedRef.current = false;
        setPreAttemptCountdownSec(null);
      }
    };
    video.addEventListener("timeupdate", onTimeUpdate);
    return () => {
      video.removeEventListener("timeupdate", onTimeUpdate);
    };
  }, [learnDrill?.step, learnDrill?.segmentId, segments]);

  useEffect(() => {
    if (learnDrill?.step !== "phrase_review") return;
    setPhraseReviewLearnKeyframes([...putTogetherKeyframesRef.current]);
  }, [learnDrill?.step, learnDrill?.segmentId]);

  const finalizeLearnPhraseReview = useCallback(() => {
    const ld = learnDrillRef.current;
    if (!ld || ld.step !== "phrase_review") return;
    const seg = ld.segment;
    const userMed = getMedianAngles(putTogetherBufferRef.current);
    const userMedConf: Record<string, number> = {};
    for (const k of Object.keys(userMed)) userMedConf[k] = 0.5;

    if (Object.keys(userMed).length < 2) {
      const note = "We couldn't capture enough pose data for full-phrase evaluation. Try once more.";
      setFullPhraseEvalResult({ pass: false, note });
      setCoachingNote(note);
    } else {
      const refMedDirect = medianAnglesForSkeletonWindow(skeletonFrames, seg.start_time_ms, seg.end_time_ms);
      const refMedSwapped = swapLeftRightJointAngles(refMedDirect);
      const useSwapped = sessionMirrorMappingRef.current === "lr_swapped";
      const refMed = useSwapped ? refMedSwapped : refMedDirect;
      const cmp = compareChunkToReference(userMed, userMedConf, refMed, PHRASE_EVAL_THRESHOLD_DEG);

      const note = cmp.pass
        ? "Great full-phrase run. You're ready to move on."
        : `Full phrase needs one more pass — biggest mismatch was ${cmp.worstLabel} (${cmp.worstDiff.toFixed(1)}deg off). Repeat the full phrase now, or go back through chunks for detail work.`;
      setFullPhraseEvalResult({ pass: cmp.pass, note });
      setCoachingNote(note);
    }

    setCompletedLearnDrillSegmentId(seg.segment_id);
    putTogetherBufferRef.current = [];
    putTogetherKeyframesRef.current = [];
    putTogetherSmoothedRef.current = null;
    setPhraseReviewLearnKeyframes([]);
    setLearnDrill((prev) => (prev && prev.step === "phrase_review" ? { ...prev, step: "phrase_complete" } : prev));
  }, [skeletonFrames]);

  // Dance: after each phrase end, optional moment review overlay
  useEffect(() => {
    if (!practiceMode || useTestVideo || practiceModeType !== "dance") return;
    const video = videoRef.current;
    if (!video) return;

    let prevT = video.currentTime * 1000;

    const onTimeUpdate = () => {
      const t = video.currentTime * 1000;
      if (phraseReviewDanceRef.current) {
        prevT = t;
        return;
      }
      if (Math.abs(t - prevT) > 2500) {
        prevT = t;
        return;
      }

      for (const seg of segments) {
        const wasInside = prevT >= seg.start_time_ms && prevT <= seg.end_time_ms;
        const crossedEnd = wasInside && prevT <= seg.end_time_ms && t > seg.end_time_ms;
        if (crossedEnd) {
          const inside = [...dancePhraseKeyframesRef.current].filter(
            (k) => k.timestamp_ms >= seg.start_time_ms && k.timestamp_ms <= seg.end_time_ms
          );
          dancePhraseKeyframesRef.current = [];
          dancePhraseSmoothedRef.current = null;
          setPhraseReviewDance({ segment: seg, keyframes: inside });
          prevT = t;
          video.pause();
          return;
        }
      }
      prevT = t;
    };

    video.addEventListener("timeupdate", onTimeUpdate);
    return () => video.removeEventListener("timeupdate", onTimeUpdate);
  }, [practiceMode, useTestVideo, practiceModeType, segments]);

  // Pose comparison + coaching trigger effect - runs every 1 second using rolling buffer median
  useEffect(() => {
    if (!practiceMode || !detectorReady || skeletonFrames.length === 0) {
      setPoseScore(null);
      setDisplayedScore(null);
      setWorstJoint(null);
      setTestDiagnostics(null);
      jointAnglesBufferRef.current = []; // Clear buffer when stopping
      jointConfidenceBufferRef.current = [];
      segmentSecondsRef.current = 0;
      return;
    }

    // Learn chunk drill uses its own evaluate path
    if (practiceModeType === "learn" && !useTestVideo && learnDrill) {
      return;
    }

    const compareInterval = setInterval(() => {
      const video = videoRef.current;
      const buffer = jointAnglesBufferRef.current;
      const confBuffer = jointConfidenceBufferRef.current;

      // Need at least 10 frames in buffer for stable comparison
      if (!video || buffer.length < 10 || confBuffer.length < 10) {
        return;
      }

      // For normal webcam mode, use the full rolling median for stability.
      // For test mode (video vs itself), use only a tiny recent window to avoid
      // lag-induced mismatch against the current reference frame.
      const userMedianAngles = useTestVideo
        ? getMedianAngles(buffer.slice(-3))
        : getMedianAngles(buffer);
      const userMedianConfidence = useTestVideo
        ? getMedianConfidence(confBuffer.slice(-3))
        : getMedianConfidence(confBuffer);
      
      if (Object.keys(userMedianAngles).length === 0) {
        return;
      }

      // Find the reference frame closest to current video time
      const timeMs = (video.currentTime ?? 0) * 1000;
      let best = 0;
      let bestDiff = Math.abs((skeletonFrames[0]?.timestamp_ms ?? 0) - timeMs);
      for (let i = 1; i < skeletonFrames.length; i++) {
        const diff = Math.abs((skeletonFrames[i]?.timestamp_ms ?? 0) - timeMs);
        if (diff < bestDiff) {
          bestDiff = diff;
          best = i;
        }
      }

      const frame = skeletonFrames[best];
      const refLandmarks = frame?.landmarks ?? [];
      if (refLandmarks.length === 0) return;

      const refAnglesDirect = useTestVideo
        ? interpolatedAnglesAtTime(skeletonFrames, timeMs)
        : calculateJointAngles(landmarkMap(refLandmarks));
      const refAnglesSwapped = swapLeftRightJointAngles(refAnglesDirect);

      // Compare median user angles to reference. In test mode, accept the better of
      // direct vs left/right-swapped mapping to absorb handedness mismatches between
      // MoveNet output and stored reference landmarks.
      const testJointConfMin = 0.45;
      const filteredUserAngles =
        useTestVideo
          ? Object.fromEntries(
              Object.entries(userMedianAngles).filter(([k, v]) => {
                const c = userMedianConfidence[k] ?? 0;
                return typeof v === "number" && c >= testJointConfMin;
              })
            )
          : userMedianAngles;
      const directCmp = comparePoses(filteredUserAngles, refAnglesDirect);
      const swappedCmp = comparePoses(filteredUserAngles, refAnglesSwapped);
      const useSwapped = sessionMirrorMappingRef.current === "lr_swapped";
      const refAngles = useSwapped ? refAnglesSwapped : refAnglesDirect;
      const { score, worstJoint: worst, worstDiff, jointsCompared } = useSwapped ? swappedCmp : directCmp;

      // Only update if we have enough joints
      if (score >= 0 && jointsCompared >= 2) {
        setPoseScore(Math.round(score));
        setWorstJoint(worst);
        if (useTestVideo) {
          const testJointConfMin = 0.45;
          const validJoints = Object.keys(userMedianAngles).filter((k) => {
            const a = userMedianAngles[k];
            const c = userMedianConfidence[k] ?? 0;
            return typeof a === "number" && a !== 0 && c >= testJointConfMin;
          });
          const confidences = validJoints
            .map((k) => userMedianConfidence[k] ?? 0)
            .filter((v) => Number.isFinite(v));
          const avgConfidence =
            confidences.length > 0
              ? confidences.reduce((sum, v) => sum + v, 0) / confidences.length
              : 0;
          setTestDiagnostics({
            score: Math.round(score),
            jointsCompared,
            worstJoint: worst,
            worstDiff,
            validJoints: validJoints.length,
            avgConfidence,
            frameOffsetMs: Math.round(bestDiff),
            mappingMode: useSwapped ? "lr_swapped" : "direct",
          });
        } else {
          setTestDiagnostics(null);
        }

        const inferredSegmentId =
          activeSegmentId ??
          segments.find(
            (s) => timeMs >= s.start_time_ms && timeMs <= s.end_time_ms
          )?.segment_id ??
          null;

        if (inferredSegmentId !== null) {
          segmentSecondsRef.current += 1;

          // Dance mode (webcam + test): silent performance log only — no real-time coaching.
          // Feedback should be session-level via End Session review.
          if (practiceModeType === "dance") {
            const perJoint: Record<string, number> = {};
            for (const { name } of JOINT_ANGLES) {
              const u = filteredUserAngles[name];
              const r = refAngles[name];
              if (u !== undefined && r !== undefined) {
                perJoint[name] = Math.abs(u - r);
              }
            }
            dancePerformanceLogRef.current.push({
              segment_id: inferredSegmentId,
              timestamp_ms: timeMs,
              per_joint_abs_diff: perJoint,
            });
            return;
          }
        } else {
          segmentSecondsRef.current = 0;
        }
      }
    }, 1000); // Check every 1 second

    return () => clearInterval(compareInterval);
  }, [
    practiceMode,
    detectorReady,
    skeletonFrames,
    useTestVideo,
    activeSegmentId,
    segments,
    coachingLoading,
    practiceModeType,
    sendCoachingRequest,
    learnDrill,
  ]);

  // Exponential moving average smoothing for displayed score (internal only)
  useEffect(() => {
    if (poseScore === null) {
      setDisplayedScore(null);
      return;
    }

    // If no displayed score yet, initialize directly
    if (displayedScore === null) {
      setDisplayedScore(poseScore);
      return;
    }

    const alpha = 0.3;
    const smoothed = Math.round(alpha * poseScore + (1 - alpha) * displayedScore);
    setDisplayedScore(smoothed);
  }, [poseScore]); // Only depend on poseScore, not displayedScore (to avoid infinite loop)

  // Initialize Web Speech API voice for coaching (en-US)
  useEffect(() => {
    if (typeof window === "undefined" || typeof window.speechSynthesis === "undefined") {
      return;
    }

    const synth = window.speechSynthesis;

    const pickVoice = () => {
      const voices = synth.getVoices ? synth.getVoices() : [];
      if (!voices || voices.length === 0) return;
      // Prefer first en-US voice; fall back to first en*; then any
      const enUs = voices.find((v) => v.lang && v.lang.toLowerCase() === "en-us");
      const enAny = voices.find((v) => v.lang && v.lang.toLowerCase().startsWith("en"));
      ttsVoiceRef.current = enUs || enAny || voices[0];
      setSpeechSupported(true);
    };

    pickVoice();
    if (typeof synth.addEventListener === "function") {
      synth.addEventListener("voiceschanged", pickVoice);
      return () => synth.removeEventListener("voiceschanged", pickVoice);
    } else {
      // older API shape
      const handler = pickVoice as unknown as () => void;
      const synthWithOnVoicesChanged = synth as unknown as {
        onvoiceschanged?: null | (() => void);
      };
      synthWithOnVoicesChanged.onvoiceschanged = handler;
      return () => {
        if (synthWithOnVoicesChanged.onvoiceschanged === handler) {
          synthWithOnVoicesChanged.onvoiceschanged = null;
        }
      };
    }
  }, []);

  // Speak new coaching notes aloud when enabled (not Dance webcam — silent session)
  useEffect(() => {
    if (
      typeof window === "undefined" ||
      typeof window.speechSynthesis === "undefined" ||
      !speechSupported ||
      !ttsEnabled ||
      !coachingNote ||
      practiceModeType === "dance"
    ) {
      return;
    }

    const synth = window.speechSynthesis;
    // Cancel any in-progress speech so notes don't queue up
    synth.cancel();

    const utterance = new SpeechSynthesisUtterance(coachingNote);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    const voice = ttsVoiceRef.current;
    if (voice) {
      utterance.voice = voice;
    }

    synth.speak(utterance);
  }, [coachingNote, speechSupported, ttsEnabled, practiceModeType, useTestVideo]);

  // Learn mode: pause reference video when new coaching note appears; auto-resume after 7s or on Continue
  const resumeFromLearnCoaching = useCallback(() => {
    if (learnResumeTimerRef.current) {
      clearTimeout(learnResumeTimerRef.current);
      learnResumeTimerRef.current = null;
    }
    videoRef.current?.play();
    setLearnPausedForCoaching(false);
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    const isChunkDrillActive =
      practiceMode &&
      practiceModeType === "learn" &&
      !!learnDrill &&
      !useTestVideo;

    // During chunk drill, we explicitly control `video` playback in the drill effects.
    // Do not auto-play here when chunk feedback is showing — keep reference video paused under the card.
    if (isChunkDrillActive) {
      if (learnResumeTimerRef.current) {
        clearTimeout(learnResumeTimerRef.current);
        learnResumeTimerRef.current = null;
      }
      setLearnPausedForCoaching(false);
      if (
        learnDrill &&
        (learnDrill.step === "phrase_prep" ||
          learnDrill.step === "phrase_review" ||
          (learnDrill.step === "chunk_feedback" && !chunkReviewPlayback))
      ) {
        video?.pause();
      }
      return;
    }
    const inLearnWithNote =
      practiceMode && practiceModeType === "learn" && !!coachingNote && !learnDrill;

    if (!inLearnWithNote) {
      if (learnResumeTimerRef.current) {
        clearTimeout(learnResumeTimerRef.current);
        learnResumeTimerRef.current = null;
      }
      video?.play();
      setLearnPausedForCoaching(false);
      return;
    }

    video?.pause();
    setLearnPausedForCoaching(true);
    if (learnResumeTimerRef.current) clearTimeout(learnResumeTimerRef.current);
    learnResumeTimerRef.current = setTimeout(() => {
      learnResumeTimerRef.current = null;
      videoRef.current?.play();
      setLearnPausedForCoaching(false);
    }, 7000);

    return () => {
      if (learnResumeTimerRef.current) {
        clearTimeout(learnResumeTimerRef.current);
        learnResumeTimerRef.current = null;
      }
    };
  }, [coachingNote, practiceMode, practiceModeType, learnDrill, useTestVideo, chunkReviewPlayback]);

  const setRate = useCallback((rate: 1 | 0.75 | 0.5) => {
    setPlaybackRate(rate);
    if (videoRef.current) {
      videoRef.current.playbackRate = rate;
    }
  }, []);

  const attemptReviewDebugRows = useMemo(() => {
    if (
      !chunkReviewPlayback ||
      !debugMode ||
      !learnDrill ||
      attemptReviewKeypoints.length === 0 ||
      skeletonFrames.length === 0
    ) {
      return null;
    }
    const chunk = learnDrill.chunks[learnDrill.chunkIndex];
    if (!chunk) return null;
    const timeMs = reviewPlaybackTimeMs;
    let bestKf = attemptReviewKeypoints[0];
    let bd = Math.abs(bestKf.timestamp_ms - timeMs);
    for (let i = 1; i < attemptReviewKeypoints.length; i++) {
      const d = Math.abs(attemptReviewKeypoints[i].timestamp_ms - timeMs);
      if (d < bd) {
        bd = d;
        bestKf = attemptReviewKeypoints[i];
      }
    }
    const userAngles = calculateJointAngles(bestKf.joints);
    const refDirect = interpolatedAnglesAtTime(skeletonFrames, timeMs);
    const refSwapped = swapLeftRightJointAngles(refDirect);
    const refAngles = sessionMirrorMapping === "lr_swapped" ? refSwapped : refDirect;

    return JOINT_ANGLES.map(({ name, label }) => {
      const u = userAngles[name];
      const r = refAngles[name];
      if (u === undefined || r === undefined) {
        return {
          name,
          label,
          user: null as number | null,
          ref: null as number | null,
          diff: null as number | null,
          highlightFocus: null as boolean | null,
        };
      }
      const diff = Math.abs(u - r);
      return {
        name,
        label,
        user: u,
        ref: r,
        diff,
        highlightFocus: chunkDrillFocusJoint === name,
      };
    });
  }, [
    chunkReviewPlayback,
    debugMode,
    learnDrill,
    attemptReviewKeypoints,
    skeletonFrames,
    reviewPlaybackTimeMs,
    sessionMirrorMapping,
    chunkDrillFocusJoint,
  ]);

  const stepReviewKeyframe = useCallback(
    (delta: number) => {
      const v = videoRef.current;
      const kfs = attemptReviewKeypoints;
      if (!v || kfs.length === 0) return;
      const t = reviewPlaybackTimeMs;
      let idx = 0;
      let bestD = Infinity;
      for (let i = 0; i < kfs.length; i++) {
        const d = Math.abs(kfs[i].timestamp_ms - t);
        if (d < bestD) {
          bestD = d;
          idx = i;
        }
      }
      const next = Math.max(0, Math.min(kfs.length - 1, idx + delta));
      v.currentTime = kfs[next].timestamp_ms / 1000;
      setReviewPlaybackTimeMs(kfs[next].timestamp_ms);
    },
    [attemptReviewKeypoints, reviewPlaybackTimeMs]
  );

  const learnDrillBigCountdown =
    practiceMode && learnDrill && !useTestVideo
      ? phrasePrepCountdownSec != null && phrasePrepCountdownSec > 0
        ? phrasePrepCountdownSec
        : preAttemptCountdownSec != null && preAttemptCountdownSec > 0
          ? preAttemptCountdownSec
          : null
      : null;

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0d0d0d] flex items-center justify-center">
        <div className="size-10 border-2 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin" />
      </div>
    );
  }

  if (error || !choreoData) {
    return (
      <div className="min-h-screen bg-[#0d0d0d] text-zinc-100 flex flex-col items-center justify-center gap-4 px-4">
        <p className="text-zinc-400">{error ?? "No data"}</p>
        <button
          type="button"
          onClick={() => router.push("/")}
          className="text-emerald-400 hover:text-emerald-300 font-medium"
        >
          Back to home
        </button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0d0d0d] text-zinc-100 flex flex-col">
      {/* Full-screen positioning guide overlay */}
      {positioningActive && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-6">
          <div className="max-w-lg w-full rounded-2xl border border-emerald-500/30 bg-zinc-950/80 p-6 shadow-xl">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold text-zinc-100">
                  Step back until your full body is visible in frame
                </h2>
                <p className="mt-2 text-sm text-zinc-300">
                  We’ll start once we can clearly detect both ankles. This improves tracking and coaching quality.
                </p>
              </div>
              <button
                type="button"
                onClick={stopPractice}
                className="inline-flex items-center justify-center h-7 w-7 rounded-full border border-zinc-600 text-zinc-300 hover:text-white hover:border-zinc-300 transition"
                aria-label="Exit practice"
              >
                <span className="sr-only">Exit practice</span>
                <svg viewBox="0 0 20 20" className="h-3.5 w-3.5" aria-hidden="true">
                  <path
                    d="M5 5l10 10M15 5L5 15"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                  />
                </svg>
              </button>
            </div>

            <div className="mt-5 flex items-center justify-center">
              {/* Simple stick figure guide */}
              <svg viewBox="0 0 100 220" className="h-56 w-auto text-emerald-400/80">
                {/* Head */}
                <circle cx="50" cy="30" r="10" fill="none" stroke="currentColor" strokeWidth="3" />
                {/* Spine */}
                <line x1="50" y1="40" x2="50" y2="120" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                {/* Arms */}
                <line x1="25" y1="70" x2="75" y2="70" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                {/* Legs */}
                <line x1="50" y1="120" x2="30" y2="190" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                <line x1="50" y1="120" x2="70" y2="190" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
              </svg>
            </div>

            {positioningWarning && (
              <div className="mt-4 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
                {positioningWarning}
              </div>
            )}
          </div>
        </div>
      )}

      {performanceReview && (
        <div className="fixed inset-0 z-[70] bg-black/90 flex flex-col items-center justify-center p-6 overflow-y-auto">
          <div className="max-w-lg w-full rounded-2xl border border-emerald-500/30 bg-zinc-950 p-6 shadow-xl">
            <div className="flex justify-between items-start gap-4 mb-4">
              <h2 className="text-lg font-semibold text-white">Session review</h2>
              <button
                type="button"
                onClick={() => setPerformanceReview(null)}
                className="text-sm text-zinc-400 hover:text-white"
              >
                Close
              </button>
            </div>
            <p className="text-zinc-200 text-sm leading-relaxed">{performanceReview.overall_assessment}</p>
            {performanceReview.top_phrases.length > 0 && (
              <div className="mt-4">
                <h3 className="text-xs font-semibold uppercase tracking-wide text-emerald-400 mb-2">
                  Phrases to work on
                </h3>
                <ul className="space-y-2 text-sm text-zinc-300">
                  {performanceReview.top_phrases.map((p, i) => (
                    <li key={i}>
                      <span className="text-emerald-300">Phrase {(p.segment_id ?? 0) + 1}</span>
                      {p.reason ? ` — ${p.reason}` : ""}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            <p className="mt-4 text-sm text-amber-200/90 border-t border-zinc-800 pt-4">
              Next focus: {performanceReview.next_session_focus}
            </p>
          </div>
        </div>
      )}

      {/* Top bar — above positioning overlay so Learn/Dance and Stop are always reachable */}
      <header className="relative z-[60] flex items-center justify-between gap-4 px-4 py-3 border-b border-zinc-800 shrink-0 bg-[#0d0d0d]">
        <button
          type="button"
          onClick={() => router.push("/")}
          className="text-sm text-zinc-500 hover:text-white transition"
        >
          ← Home
        </button>
        <div className="flex items-center gap-4">
          {/* Practice mode button */}
          {!practiceMode ? (
            <>
              <button
                type="button"
                onClick={() => startPractice(false)}
                disabled={webcamLoading}
                className="rounded-lg px-3 py-1.5 text-xs font-medium transition bg-red-500/20 text-red-400 hover:bg-red-500/30"
              >
                {webcamLoading ? "Starting..." : "Start Practice"}
              </button>
              <button
                type="button"
                onClick={() => startPractice(true)}
                disabled={webcamLoading}
                className="rounded-lg px-3 py-1.5 text-xs font-medium transition bg-purple-500/20 text-purple-400 hover:bg-purple-500/30"
                title="Use reference video as input to test scoring accuracy"
              >
                Test Mode
              </button>
            </>
          ) : (
            <>
              {practiceModeType === "dance" && (
                <button
                  type="button"
                  onClick={endDanceSession}
                  disabled={performanceReviewLoading}
                  className="rounded-lg px-3 py-1.5 text-xs font-medium transition bg-amber-500/20 text-amber-300 hover:bg-amber-500/30 disabled:opacity-50"
                >
                  {performanceReviewLoading ? "Review…" : "End Session"}
                </button>
              )}
              <button
                type="button"
                onClick={stopPractice}
                className="rounded-lg px-3 py-1.5 text-xs font-medium transition bg-zinc-700 text-white hover:bg-zinc-600"
              >
                Stop Practice
              </button>
            </>
          )}
          <button
            type="button"
            onClick={() => setShowSkeleton((v) => !v)}
            disabled={skeletonLoading}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
              showSkeleton
                ? "bg-emerald-500/20 text-emerald-400"
                : "bg-zinc-900 text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {skeletonLoading ? "Loading skeleton…" : "Show Reference Skeleton"}
          </button>
          <button
            type="button"
            onClick={() => setIsReferenceMirrored((v) => !v)}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
              isReferenceMirrored
                ? "bg-cyan-500/20 text-cyan-300"
                : "bg-zinc-900 text-zinc-500 hover:text-zinc-300"
            }`}
            title="Mirror the reference video and keep replay orientation aligned"
          >
            {isReferenceMirrored ? "Reference Mirrored" : "Mirror Reference"}
          </button>
          <button
            type="button"
            onClick={() => setDebugMode((d) => !d)}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
              debugMode
                ? "bg-amber-500/25 text-amber-300 border border-amber-500/50"
                : "bg-zinc-900 text-zinc-500 hover:text-zinc-300 border border-zinc-700"
            }`}
            title="Show attempt playback angle debug table and overlays"
          >
            {debugMode ? "Debug mode on" : "Debug mode off"}
          </button>
          <QualityBadge score={overallScore} />
          <div className="flex items-center gap-2">
            {/* Learn / Dance mode toggle — always visible; active only in practice with webcam */}
            <div
              className="flex items-center gap-1 rounded-lg bg-zinc-900 p-1"
              title={
                practiceMode
                  ? "Learn (slow, detailed) or Dance (full speed) — works in practice and test mode"
                  : "Start Practice or Test Mode to switch Learn / Dance"
              }
            >
              {(["learn", "dance"] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  disabled={!practiceMode}
                  onClick={() => {
                    if (!practiceMode) return;
                    setPracticeModeType(mode);
                    setSuggestDanceMode(false);
                    if (mode === "dance") {
                      setShowSkeleton(true);
                      setLearnDrill(null);
                      learnDrillRef.current = null;
                      learnCaptureRef.current = "none";
                      setPreAttemptCountdownSec(null);
                    }
                    const rate = mode === "learn" ? 0.5 : 1;
                    setPlaybackRate(rate);
                    if (videoRef.current) videoRef.current.playbackRate = rate;
                  }}
                  className={`px-2.5 py-1 text-xs font-medium rounded transition capitalize ${
                    !practiceMode
                      ? "text-zinc-600 cursor-not-allowed"
                      : practiceModeType === mode
                        ? "bg-emerald-500/20 text-emerald-300"
                        : "text-zinc-500 hover:text-zinc-300"
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-1 rounded-lg bg-zinc-900 p-1">
              {([1, 0.75, 0.5] as const).map((rate) => (
                <button
                  key={rate}
                  type="button"
                  onClick={() => setRate(rate)}
                  className={`px-2.5 py-1 text-xs font-medium rounded transition ${
                    playbackRate === rate
                      ? "bg-zinc-700 text-white"
                      : "text-zinc-500 hover:text-zinc-300"
                  }`}
                >
                  {rate}x
                </button>
              ))}
            </div>
            {/* Coaching voice toggle */}
            <button
              type="button"
              onClick={() => setTtsEnabled((v) => !v)}
              disabled={!speechSupported}
              className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium border transition ${
                !speechSupported
                  ? "border-zinc-700 text-zinc-600 cursor-not-allowed"
                  : ttsEnabled
                  ? "border-emerald-500/60 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20"
                  : "border-zinc-700 bg-zinc-900 text-zinc-400 hover:text-zinc-200 hover:border-zinc-500"
              }`}
            >
              <svg viewBox="0 0 20 20" className="h-3.5 w-3.5" aria-hidden="true">
                {ttsEnabled ? (
                  <>
                    <path d="M3 8.5h3.2L9 5.5v9l-2.8-3H3v-3z" fill="currentColor" />
                    <path
                      d="M13 6.5a3 3 0 010 7"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.6"
                      strokeLinecap="round"
                    />
                  </>
                ) : (
                  <>
                    <path d="M3 8.5h3.2L9 5.5v9l-2.8-3H3v-3z" fill="currentColor" />
                    <path
                      d="M13 6l4 8M17 6l-4 8"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.6"
                      strokeLinecap="round"
                    />
                  </>
                )}
              </svg>
              <span>{ttsEnabled ? "Voice on" : "Voice off"}</span>
            </button>
            <div className="flex items-center gap-2 rounded-lg bg-zinc-900/70 border border-zinc-800 px-3 py-1.5">
              {authLoading ? (
                <span className="text-[11px] text-zinc-600">Loading…</span>
              ) : userEmail ? (
                <>
                  <span className="text-[11px] text-zinc-300 max-w-[180px] truncate">
                    {userEmail}
                  </span>
                  <button
                    type="button"
                    onClick={handleSignOut}
                    disabled={signOutLoading}
                    className={`px-2.5 py-1 text-[11px] font-medium rounded transition border ${
                      signOutLoading
                        ? "bg-zinc-800 border-zinc-700 text-zinc-400 cursor-not-allowed"
                        : "bg-zinc-700 border-zinc-500 text-white hover:bg-zinc-600"
                    }`}
                  >
                    {signOutLoading ? "Signing out..." : "Sign out"}
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  onClick={() => router.push(`/auth/login?next=${encodeURIComponent(videoId ? `/player/${videoId}` : "/")}`)}
                  disabled={authLoading}
                  className={`px-2.5 py-1 text-[11px] font-medium rounded transition border ${
                    authLoading
                      ? "bg-zinc-800 border-zinc-700 text-zinc-400 cursor-not-allowed"
                      : "bg-zinc-700 border-zinc-500 text-white hover:bg-zinc-600"
                  }`}
                >
                  Sign in
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {webcamError && (
        <div className="px-4 py-2 bg-red-500/10 border-b border-red-500/30 text-red-400 text-sm">
          {webcamError}
        </div>
      )}

      <div className="flex flex-1 min-h-0">
        <div className="flex-1 min-h-0 min-w-0 flex flex-col gap-2 p-2 sm:p-4 min-h-0 overflow-hidden">
          <div className="flex flex-1 min-h-0 gap-2 relative min-h-0">
          {/* Left: Reference video */}
          <div className={`${practiceMode && !useTestVideo ? "w-1/2" : "w-full"} h-full flex items-center justify-center transition-all`}>
            <div
              ref={videoContainerRef}
              className="w-full h-full max-w-full max-h-full aspect-video rounded-xl overflow-hidden bg-black relative"
            >
              <video
                ref={videoRef}
                src={`${API_URL}/video/${jobId}`}
                crossOrigin="anonymous"
                className={`absolute inset-0 w-full h-full object-contain ${isReferenceMirrored ? "scale-x-[-1]" : ""}`}
                controls
                playsInline
                onLoadedMetadata={() => {
                  if (videoRef.current) {
                    videoRef.current.playbackRate = playbackRate;
                  }
                }}
                onEnded={stopLoop}
              />
              {showSkeleton && (
                <>
                  <div
                    className="absolute inset-0 w-full h-full pointer-events-none rounded-xl bg-black/40"
                    style={{ zIndex: 5 }}
                  />
                  <canvas
                    ref={canvasRef}
                    className="absolute inset-0 w-full h-full pointer-events-none rounded-xl"
                    style={{ zIndex: 10 }}
                  />
                </>
              )}
              {practiceMode && (
                <div className="absolute top-2 left-2 bg-black/60 rounded px-2 py-1 text-xs text-emerald-400 z-20">
                  Reference
                </div>
              )}
              {practiceMode && useTestVideo && testDiagnostics && (
                <div className="absolute top-2 right-2 z-20 rounded-lg border border-cyan-400/40 bg-black/70 px-3 py-2 text-[11px] text-cyan-100 shadow-lg backdrop-blur">
                  <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-cyan-300">
                    Test diagnostics
                  </div>
                  <div>Score: {testDiagnostics.score}%</div>
                  <div>Joints compared: {testDiagnostics.jointsCompared}</div>
                  <div>
                    Worst: {testDiagnostics.worstJoint ?? "none"} ({testDiagnostics.worstDiff.toFixed(1)}deg)
                  </div>
                  <div>
                    Valid joints: {testDiagnostics.validJoints} | Avg conf:{" "}
                    {(testDiagnostics.avgConfidence * 100).toFixed(0)}%
                  </div>
                  <div>Frame offset: {testDiagnostics.frameOffsetMs}ms</div>
                  <div>Mapping: {testDiagnostics.mappingMode === "lr_swapped" ? "L/R swapped" : "Direct"}</div>
                </div>
              )}
              {learnDrillBigCountdown != null && (
                <div className="absolute inset-0 z-[25] flex items-center justify-center rounded-xl bg-black/50 pointer-events-none">
                  <span className="text-7xl font-black tabular-nums text-white drop-shadow-lg">
                    {learnDrillBigCountdown}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Center: Learn chunk drill OR coaching (test / dance has no live coaching text) */}
          {practiceMode && learnDrill && !useTestVideo ? (
            learnFeedbackPanelVisible ? (
            <div className="absolute left-1/2 -translate-x-1/2 top-4 z-30 flex flex-col items-center pointer-events-none gap-2 w-full max-w-xl px-2">
              <div className="pointer-events-auto rounded-xl bg-black/80 border border-emerald-500/40 shadow-lg w-full px-5 py-4 text-left">
                <div className="text-xs font-semibold uppercase tracking-wide text-emerald-300 mb-2">
                  {learnDrill.step === "phrase_prep"
                    ? "Learn drill — full phrase next"
                    : `Learn drill — chunk ${learnDrill.chunkIndex + 1} / ${learnDrill.chunks.length}`}
                </div>
                {learnDrill.step === "preview" && (
                  <>
                    <p className="text-2xl font-semibold text-white mb-2">Watch</p>
                    <p className="text-sm text-zinc-300 mb-4">
                      Preview this slice at half speed with the reference skeleton. Webcam stays off until your turn.
                    </p>
                    {preAttemptCountdownSec !== null && preAttemptCountdownSec > 0 ? (
                      <p className="text-center text-sm text-emerald-200 py-2">
                        Get ready — recording starts after the countdown.
                      </p>
                    ) : (
                      <button
                        type="button"
                        onClick={startChunkAttemptFromPreview}
                        className="w-full rounded-lg px-3 py-2 text-sm font-medium bg-emerald-500 text-white hover:bg-emerald-400 transition"
                      >
                        Start your turn
                      </button>
                    )}
                  </>
                )}
                {learnDrill.step === "attempt" && (
                  <>
                    <p className="text-2xl font-semibold text-white mb-1">Your turn</p>
                    <p className="text-sm text-zinc-300">
                      Match the reference.{" "}
                      <span className="text-emerald-300 font-mono">
                        {attemptCountdownSec != null ? `${attemptCountdownSec}s` : "—"}
                      </span>{" "}
                      left
                    </p>
                  </>
                )}
                {learnDrill.step === "chunk_feedback" && (
                  <>
                    <p className="text-xs text-zinc-500 mb-2">
                      Attempt {(learnDrill.chunkAttemptsOnSlice ?? 0)} of 3 on this slice
                      {(learnDrill.chunkAttemptsOnSlice ?? 0) >= 3 ? " — moving on automatically…" : ""}
                    </p>
                    <p className="text-lg text-zinc-100 mb-3">
                      {coachingLoading ? "Getting a coaching tip…" : coachingNote ?? "Review your replay below — yellow highlights the joint with the biggest gap vs the reference."}
                    </p>
                    <div className="mb-4 flex flex-wrap items-center gap-2">
                      {attemptReviewKeypoints.length > 0 ? (
                        <>
                          <button
                            type="button"
                            onClick={() => setChunkReviewPlayback((v) => !v)}
                            className={`rounded-lg px-3 py-2 text-sm font-medium transition border ${
                              chunkReviewPlayback
                                ? "bg-emerald-600 border-emerald-400 text-white"
                                : "bg-zinc-800 border-zinc-600 text-zinc-100 hover:bg-zinc-700"
                            }`}
                          >
                            Review my attempt
                          </button>
                          {chunkReviewPlayback && (
                            <button
                              type="button"
                              onClick={() => setChunkReviewPlayback(false)}
                              className="rounded-lg px-3 py-2 text-sm font-medium bg-zinc-700 text-white border border-zinc-500 hover:bg-zinc-600 transition"
                            >
                              Back to camera
                            </button>
                          )}
                        </>
                      ) : (
                        <p className="text-xs text-zinc-500">
                          No pose replay — we didn&apos;t capture enough skeleton frames. Stay in frame for the full chunk and ensure lighting lets us see your body.
                        </p>
                      )}
                    </div>
                    <div className="flex flex-col gap-2 sm:flex-row">
                      {preAttemptCountdownSec !== null && preAttemptCountdownSec > 0 ? (
                        <p className="text-center text-sm text-emerald-200 py-2 sm:flex-1">
                          Get ready — recording starts after the countdown.
                        </p>
                      ) : (
                        <>
                          <button
                            type="button"
                            onClick={retryChunkFromFeedback}
                            disabled={(learnDrill.chunkAttemptsOnSlice ?? 0) >= 3}
                            className="sm:flex-1 rounded-lg px-3 py-2 text-sm font-medium bg-zinc-700 text-white hover:bg-zinc-600 transition disabled:opacity-40 disabled:pointer-events-none"
                          >
                            Try again
                          </button>
                          <button
                            type="button"
                            onClick={moveToNextChunkFromFeedback}
                            className="sm:flex-1 rounded-lg px-3 py-2 text-sm font-medium bg-emerald-500 text-white hover:bg-emerald-400 transition"
                          >
                            Move on
                          </button>
                        </>
                      )}
                    </div>
                  </>
                )}
                {learnDrill.step === "phrase_prep" && (
                  <>
                    <p className="text-2xl font-semibold text-amber-200 mb-2">Get ready</p>
                    <p className="text-sm text-zinc-300">
                      Full phrase runs at full speed next. Step into frame and find your starting position — we&apos;ll
                      count down from the number on the video.
                    </p>
                  </>
                )}
                {learnDrill.step === "put_together" && (
                  <p className="text-xl font-semibold text-amber-200">Put it together — full speed, end to end</p>
                )}
                {learnDrill.step === "phrase_review" && (
                  <p className="text-sm text-zinc-400">Review your phrase highlights in the overlay above.</p>
                )}
                {learnDrill.step === "phrase_complete" && (
                  <>
                    <p className={`text-lg mb-3 ${fullPhraseEvalResult?.pass ? "text-emerald-300" : "text-zinc-100"}`}>
                      {fullPhraseEvalResult?.note ?? "Full-phrase evaluation complete."}
                    </p>
                    <div className="flex flex-col gap-2 sm:flex-row">
                      <button
                        type="button"
                        onClick={() => {
                          if (!learnDrillRef.current) return;
                          setCompletedLearnDrillSegmentId(null);
                          setFullPhraseEvalResult(null);
                          setCoachingNote(null);
                          setPhrasePrepCountdownSec(null);
                          setLearnDrill((prev) =>
                            prev ? { ...prev, step: "phrase_prep", chunkAttemptsOnSlice: 0 } : prev
                          );
                        }}
                        className="sm:flex-1 rounded-lg px-3 py-2 text-sm font-medium bg-emerald-500 text-white hover:bg-emerald-400 transition"
                      >
                        Repeat full phrase
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          if (!learnDrillRef.current) return;
                          setCompletedLearnDrillSegmentId(null);
                          setFullPhraseEvalResult(null);
                          setCoachingNote(null);
                          chunkFeedbackProcessedKeyRef.current = "";
                          setPhrasePrepCountdownSec(null);
                          setLearnDrill((prev) =>
                            prev
                              ? { ...prev, chunkIndex: 0, step: "preview", chunkAttemptsOnSlice: 0 }
                              : prev
                          );
                        }}
                        className="sm:flex-1 rounded-lg px-3 py-2 text-sm font-medium bg-zinc-700 text-white hover:bg-zinc-600 transition"
                      >
                        Redo chunks
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
            ) : null
          ) : (
            practiceMode && (
              <div className="absolute left-1/2 -translate-x-1/2 top-4 z-30 flex flex-col items-center pointer-events-none gap-2">
                {suggestDanceMode && practiceModeType === "learn" && !learnDrill && (
                  <div className="pointer-events-auto rounded-xl bg-emerald-500/20 border border-emerald-400/50 px-4 py-3 shadow-lg flex items-center gap-3">
                    <span className="text-sm text-emerald-100">Ready to try at full speed?</span>
                    <button
                      type="button"
                      onClick={() => {
                        setPracticeModeType("dance");
                        setShowSkeleton(true);
                        setSuggestDanceMode(false);
                        setPlaybackRate(1);
                        if (videoRef.current) videoRef.current.playbackRate = 1;
                      }}
                      className="rounded-lg px-3 py-1.5 text-sm font-medium bg-emerald-500 text-white hover:bg-emerald-400 transition"
                    >
                      Switch to Dance
                    </button>
                  </div>
                )}
                {practiceModeType === "dance" ? (
                  <div className="pointer-events-auto rounded-xl bg-black/70 border border-zinc-600/50 shadow-lg max-w-sm w-full px-4 py-3">
                    <span className="text-xs font-semibold uppercase tracking-wide text-zinc-400">Dance mode</span>
                    <p className="text-sm text-zinc-300 mt-1">
                      No live coaching — we log your joint match quietly. Use End Session for a full review.
                      {useTestVideo ? " Test mode compares the reference video against itself." : ""}
                    </p>
                  </div>
                ) : (
                  <div
                    className={`pointer-events-auto rounded-xl bg-black/70 border border-emerald-500/30 shadow-lg w-full transition-all duration-300 ${
                      practiceModeType === "learn" ? "px-5 py-4 max-w-xl text-left" : "px-4 py-3 max-w-sm"
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <div
                        className={`h-2 w-2 rounded-full ${coachingLoading ? "bg-emerald-400 animate-pulse" : "bg-emerald-400/60"}`}
                      />
                      <span className="text-xs font-semibold uppercase tracking-wide text-emerald-300">
                        Coaching{useTestVideo ? " (test)" : ""}
                      </span>
                    </div>
                    <p
                      className={`text-zinc-100 transition-opacity duration-500 ${
                        practiceModeType === "learn" ? "text-lg md:text-xl" : "text-sm"
                      } ${coachingNote ? "opacity-100" : "opacity-70"}`}
                    >
                      {coachingNote ??
                        (useTestVideo
                          ? "Play a phrase — feedback uses the same pipeline with the reference video as pose input (expect near-perfect match)."
                          : "Hold this phrase for a few seconds while we watch your lines, then coaching tips will appear here.")}
                    </p>
                    {worstJoint && coachingNote && (
                      <p className={`mt-1 text-emerald-200/80 ${practiceModeType === "learn" ? "text-sm" : "text-xs"}`}>
                        Focus area: <span className="font-medium capitalize">{worstJoint}</span>
                      </p>
                    )}
                    {practiceModeType === "learn" && learnPausedForCoaching && (
                      <button
                        type="button"
                        onClick={resumeFromLearnCoaching}
                        className="mt-3 w-full rounded-lg px-3 py-2 text-sm font-medium bg-emerald-500 text-white hover:bg-emerald-400 transition"
                      >
                        Continue
                      </button>
                    )}
                  </div>
                )}
              </div>
            )
          )}

          {/* Right: Webcam panel (only in webcam mode, not test mode) */}
          {practiceMode && !useTestVideo && (
            <div className="w-1/2 h-full flex items-center justify-center">
              <div
                ref={webcamContainerRef}
                className={`w-full max-h-full aspect-video rounded-xl overflow-hidden bg-zinc-900 relative transition-opacity ${
                  learnDrill?.step === "preview" ? "opacity-0 pointer-events-none" : "opacity-100"
                }`}
              >
                {/* Offscreen video element for webcam stream (must not be display:none for canvas to read) */}
                <video
                  ref={webcamRef}
                  className="absolute w-0 h-0 opacity-0"
                  playsInline
                  muted
                  autoPlay
                />
                {/* Canvas for webcam + skeleton overlay */}
                <canvas
                  ref={webcamCanvasRef}
                  className="absolute inset-0 w-full h-full"
                  style={{ zIndex: 10 }}
                />
                <div className="absolute top-2 left-2 bg-black/60 rounded px-2 py-1 text-xs z-20 text-red-400">
                  You
                </div>
                {!detectorReady && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="size-8 border-2 border-red-500/30 border-t-red-500 rounded-full animate-spin" />
                  </div>
                )}
              </div>
            </div>
          )}
          </div>

          {practiceMode &&
            learnDrill &&
            !useTestVideo &&
            chunkReviewPlayback &&
            !learnFeedbackPanelVisible &&
            learnDrill.step === "chunk_feedback" && (
              <div className="shrink-0 flex justify-center px-2 pb-1 pt-0.5 border-t border-zinc-800/80 bg-[#0d0d0d]/95">
                <button
                  type="button"
                  onClick={() => setLearnFeedbackPanelVisible(true)}
                  className="rounded-lg border border-emerald-500/45 bg-zinc-900 px-4 py-2 text-sm font-medium text-emerald-200 hover:bg-emerald-500/15 transition shadow-md"
                >
                  Show feedback & actions
                </button>
              </div>
            )}

          {chunkReviewPlayback &&
            debugMode &&
            learnDrill &&
            attemptReviewDebugRows &&
            practiceMode &&
            !useTestVideo && (
              <div className="shrink-0 rounded-xl border border-amber-500/40 bg-zinc-950/95 p-3 max-h-[40vh] overflow-auto">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                  <span className="text-xs font-semibold uppercase tracking-wide text-amber-400">
                    Attempt playback debug
                  </span>
                  <span className="text-[11px] font-mono text-zinc-500">
                    t = {(reviewPlaybackTimeMs / 1000).toFixed(2)}s · phrase threshold {PHRASE_EVAL_THRESHOLD_DEG}° · mirror{" "}
                    {sessionMirrorMapping === "lr_swapped" ? "swapped" : "direct"}
                    {chunkDrillFocusJoint ? ` · focus ${chunkDrillFocusJoint}` : ""}
                  </span>
                </div>
                <div className="flex flex-wrap items-center gap-2 mb-3">
                  <button
                    type="button"
                    onClick={() => stepReviewKeyframe(-1)}
                    className="rounded px-2 py-1 text-xs bg-zinc-800 text-zinc-200 hover:bg-zinc-700"
                  >
                    ◀ Frame
                  </button>
                  <button
                    type="button"
                    onClick={() => stepReviewKeyframe(1)}
                    className="rounded px-2 py-1 text-xs bg-zinc-800 text-zinc-200 hover:bg-zinc-700"
                  >
                    Frame ▶
                  </button>
                  <label className="flex flex-1 items-center gap-2 min-w-[200px] text-xs text-zinc-400">
                    Scrub
                    <input
                      type="range"
                      className="flex-1 accent-amber-500"
                      min={learnDrill.chunks[learnDrill.chunkIndex].startMs}
                      max={learnDrill.chunks[learnDrill.chunkIndex].endMs}
                      step={16}
                      value={Math.min(
                        Math.max(reviewPlaybackTimeMs, learnDrill.chunks[learnDrill.chunkIndex].startMs),
                        learnDrill.chunks[learnDrill.chunkIndex].endMs
                      )}
                      onChange={(e) => {
                        const ms = Number(e.target.value);
                        const v = videoRef.current;
                        if (v) v.currentTime = ms / 1000;
                        setReviewPlaybackTimeMs(ms);
                      }}
                    />
                  </label>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-[11px] border-collapse">
                    <thead>
                      <tr className="border-b border-zinc-700 text-zinc-500">
                        <th className="py-1 pr-2 font-medium">Joint</th>
                        <th className="py-1 pr-2 font-medium">Your angle</th>
                        <th className="py-1 pr-2 font-medium">Reference angle</th>
                        <th className="py-1 pr-2 font-medium">Diff</th>
                        <th className="py-1 font-medium">Highlight</th>
                      </tr>
                    </thead>
                    <tbody>
                      {attemptReviewDebugRows.map((row) => (
                        <tr
                          key={row.name}
                          className={`border-b border-zinc-800/80 ${
                            chunkDrillFocusJoint === row.name ? "bg-amber-500/10" : ""
                          }`}
                        >
                          <td className="py-1 pr-2 text-zinc-300 capitalize">{row.label}</td>
                          <td className="py-1 pr-2 font-mono text-zinc-200">
                            {row.user != null ? row.user.toFixed(1) : "—"}
                          </td>
                          <td className="py-1 pr-2 font-mono text-zinc-200">
                            {row.ref != null ? row.ref.toFixed(1) : "—"}
                          </td>
                          <td className="py-1 pr-2 font-mono text-zinc-200">
                            {row.diff != null ? row.diff.toFixed(1) : "—"}
                          </td>
                          <td className="py-1">
                            {row.highlightFocus === null ? (
                              <span className="text-zinc-600">—</span>
                            ) : row.highlightFocus ? (
                              <span className="text-amber-300">Focus</span>
                            ) : (
                              <span className="text-zinc-500">—</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
        </div>

        {/* Segments sidebar */}
        <aside className="w-72 shrink-0 border-l border-zinc-800 flex flex-col overflow-hidden">
          <div className="px-3 py-2 border-b border-zinc-800">
            <h2 className="text-sm font-medium text-zinc-400">Phrases</h2>
          </div>
          <div className="flex-1 overflow-y-auto p-2 space-y-2">
            {segments.map((seg) => (
              <button
                key={seg.segment_id}
                type="button"
                onClick={() => playSegment(seg)}
                className={`w-full text-left rounded-lg px-3 py-2.5 transition border ${
                  activeSegmentId === seg.segment_id
                    ? "bg-emerald-500/15 border-emerald-500/50 text-white"
                    : "bg-zinc-900/80 border-zinc-800 text-zinc-300 hover:border-zinc-700 hover:bg-zinc-800/80"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-mono text-zinc-500">
                    {msToTimestamp(seg.start_time_ms)} – {msToTimestamp(seg.end_time_ms)}
                  </span>
                  <span className="text-xs text-zinc-500">{seg.beat_count} beats</span>
                </div>
                <div className="mt-0.5 text-sm font-medium">Phrase {seg.segment_id + 1}</div>
              </button>
            ))}
          </div>
        </aside>
      </div>

      {learnDrill?.step === "phrase_review" && learnDrill && (
        <PhraseReview
          jobId={jobId}
          segment={{
            segment_id: learnDrill.segment.segment_id,
            start_time_ms: learnDrill.segment.start_time_ms,
            end_time_ms: learnDrill.segment.end_time_ms,
          }}
          skeletonFrames={skeletonFrames}
          userKeyframes={phraseReviewLearnKeyframes}
          mirrorMapping={sessionMirrorMapping === "lr_swapped" ? "lr_swapped" : "direct"}
          practiceModeSide="learn"
          videoRef={videoRef}
          isReferenceMirrored={isReferenceMirrored}
          onGotIt={finalizeLearnPhraseReview}
        />
      )}

      {phraseReviewDance && (
        <PhraseReview
          jobId={jobId}
          segment={{
            segment_id: phraseReviewDance.segment.segment_id,
            start_time_ms: phraseReviewDance.segment.start_time_ms,
            end_time_ms: phraseReviewDance.segment.end_time_ms,
          }}
          skeletonFrames={skeletonFrames}
          userKeyframes={phraseReviewDance.keyframes}
          mirrorMapping={sessionMirrorMapping === "lr_swapped" ? "lr_swapped" : "direct"}
          practiceModeSide="dance"
          videoRef={videoRef}
          isReferenceMirrored={isReferenceMirrored}
          onGotIt={() => {
            setPhraseReviewDance(null);
            void videoRef.current?.play();
          }}
        />
      )}
    </div>
  );
}
