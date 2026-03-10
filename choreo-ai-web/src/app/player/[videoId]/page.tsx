"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useSearchParams, useRouter } from "next/navigation";

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
  fillColor: string
) {
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  for (const [a, b] of POSE_EDGES) {
    const pa = joints[a];
    const pb = joints[b];
    if (!pa || !pb) continue;
    const ca = toCanvas(pa);
    const cb = toCanvas(pb);
    ctx.beginPath();
    ctx.moveTo(ca.x, ca.y);
    ctx.lineTo(cb.x, cb.y);
    ctx.stroke();
  }

  ctx.fillStyle = fillColor;
  const jointRadius = 4;
  for (const name of POSE_JOINTS) {
    const p = joints[name];
    if (!p) continue;
    const c = toCanvas(p);
    ctx.beginPath();
    ctx.arc(c.x, c.y, jointRadius, 0, Math.PI * 2);
    ctx.fill();
  }
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
  const [activeSegmentId, setActiveSegmentId] = useState<number | null>(null);
  const [showSkeleton, setShowSkeleton] = useState(false);
  const [skeletonFrames, setSkeletonFrames] = useState<SkeletonFrame[]>([]);
  const [skeletonLoading, setSkeletonLoading] = useState(false);

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

  // Coaching state
  const [coachingNote, setCoachingNote] = useState<string | null>(null);
  const [coachingLoading, setCoachingLoading] = useState(false);

  // Positioning gate (preflight) before starting practice
  const [positioningActive, setPositioningActive] = useState(false);
  const [positioningWarning, setPositioningWarning] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const loopIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const videoContainerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

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
  }, []);

  const playSegment = useCallback(
    (seg: Segment) => {
      const video = videoRef.current;
      if (!video) return;

      const startSec = seg.start_time_ms / 1000;
      const endSec = seg.end_time_ms / 1000;
      video.currentTime = startSec;
      video.play();
      setActiveSegmentId(seg.segment_id);

      if (loopIntervalRef.current) clearInterval(loopIntervalRef.current);
      loopIntervalRef.current = setInterval(() => {
        if (video.currentTime >= endSec - 0.1) {
          video.currentTime = startSec;
        }
      }, 200);
    },
    []
  );

  useEffect(() => {
    return () => {
      if (loopIntervalRef.current) clearInterval(loopIntervalRef.current);
    };
  }, []);

  // Draw reference skeleton overlay synced to video time
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = videoContainerRef.current;
    const video = videoRef.current;
    if (!canvas || !container || !video || !showSkeleton || skeletonFrames.length === 0) return;

    let rafId: number;

    const draw = () => {
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
        x: offsetX + p.x * renderedWidth,
        y: offsetY + p.y * renderedHeight,
      });

      drawSkeleton(ctx, joints, toCanvas, "#00FF88", "#FFFFFF");

      rafId = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(rafId);
  }, [showSkeleton, skeletonFrames]);

  // Start practice mode: init webcam (or test video) and MoveNet
  const startPractice = useCallback(async (testMode = false) => {
    if (practiceMode) return;
    
    setWebcamLoading(true);
    setWebcamError(null);
    setUseTestVideo(testMode);
    setPositioningWarning(null);
    positioningOkRef.current = false;

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
      
      // Set practice mode first so the video element renders
      setPracticeMode(true);
      if (!testMode) {
        setPositioningActive(true);
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
  }, [practiceMode]);
  
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
  }, []);

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

        // Draw mirrored webcam feed
        ctx.save();
        ctx.translate(containerWidth, offsetY);
        ctx.scale(-1, 1);
        ctx.drawImage(webcam, 0, 0, renderedWidth, renderedHeight);
        ctx.restore();

        // Draw skeleton from latest detection results
        if (latestJoints && Object.keys(latestJoints).length > 0) {
          const toCanvas = (p: { x: number; y: number }) => ({
            x: offsetX + p.x * renderedWidth,
            y: offsetY + p.y * renderedHeight,
          });

          ctx.strokeStyle = "#FF6B6B";
          ctx.lineWidth = 4;
          ctx.lineCap = "round";
          
          for (const [a, b] of POSE_EDGES) {
            const pa = latestJoints[a];
            const pb = latestJoints[b];
            if (!pa || !pb) continue;
            const ca = toCanvas(pa);
            const cb = toCanvas(pb);
            ctx.beginPath();
            ctx.moveTo(ca.x, ca.y);
            ctx.lineTo(cb.x, cb.y);
            ctx.stroke();
          }

          ctx.fillStyle = "#FFFFFF";
          for (const name of POSE_JOINTS) {
            const p = latestJoints[name];
            if (!p) continue;
            const c = toCanvas(p);
            ctx.beginPath();
            ctx.arc(c.x, c.y, 6, 0, Math.PI * 2);
            ctx.fill();
          }
        }

        webcamRafRef.current = requestAnimationFrame(drawLoop);
      }

      async function runPoseDetection() {
        if (!running || isDetecting) return;
        
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

          // If we lose ankles for a while during practice, re-activate positioning gate
          const leftAnkleScore = data[15 * 3 + 2];
          const rightAnkleScore = data[16 * 3 + 2];
          if (leftAnkleScore > 0.3 && rightAnkleScore > 0.3) {
            lostBodyCountRef.current = 0;
          } else {
            lostBodyCountRef.current += 1;
          }

          // After ~0.5s of missing ankles, pause tracking and ask user to reposition again
          if (!positioningActive && detectorReady && lostBodyCountRef.current >= 15) {
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
  }, [practiceMode, detectorReady, useTestVideo]);

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

  // Send coaching requests to backend (throttled)
  const sendCoachingRequest = useCallback(
    async (
      segmentId: number,
      userMedianAngles: Record<string, number>,
      userMedianConfidence: Record<string, number>,
      validJoints: string[],
      matchLevel: "good" | "developing" | "needs_work"
    ) => {
      if (!jobId) return;

      const segment = segments.find((s) => s.segment_id === segmentId);
      const reference_angle_summary = segment?.angle_summary ?? {};

      try {
        setCoachingLoading(true);

        const res = await fetch(`${API_URL}/coaching`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            segment_id: segmentId,
            reference_angle_summary,
            user_angles: userMedianAngles,
            user_joint_confidence: userMedianConfidence,
            valid_joints: validJoints,
            match_level: matchLevel,
            skill_level: "beginner",
            style: "contemporary",
          }),
        });

        if (!res.ok) {
          return;
        }

        const data = (await res.json()) as { note?: string };
        if (data.note) {
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

  // Pose comparison + coaching trigger effect - runs every 1 second using rolling buffer median
  useEffect(() => {
    if (!practiceMode || !detectorReady || skeletonFrames.length === 0) {
      setPoseScore(null);
      setDisplayedScore(null);
      setWorstJoint(null);
      jointAnglesBufferRef.current = []; // Clear buffer when stopping
      jointConfidenceBufferRef.current = [];
      segmentSecondsRef.current = 0;
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

      // Get median angles from the rolling buffer
      const userMedianAngles = getMedianAngles(buffer);
      const userMedianConfidence = getMedianConfidence(confBuffer);
      
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

      const refJoints = landmarkMap(refLandmarks);
      const refAngles = calculateJointAngles(refJoints);

      // Compare median user angles to reference
      const { score, worstJoint: worst, jointsCompared } = comparePoses(userMedianAngles, refAngles);

      // Only update if we have enough joints
      if (score >= 0 && jointsCompared >= 2) {
        setPoseScore(Math.round(score));
        setWorstJoint(worst);

         // Coaching trigger (only in real practice mode, not test mode)
        if (!useTestVideo && activeSegmentId !== null) {
          // Track how long we've been in this segment
          segmentSecondsRef.current += 1; // compareInterval runs every 1s

          const now = Date.now();
          const timeSinceLast = now - lastCoachingTimeRef.current;

          if (
            segmentSecondsRef.current >= 5 && // at least 5s in this phrase
            timeSinceLast >= 8000 && // throttle: max every 8s
            !coachingLoading
          ) {
            // Reset dwell timer and update last coaching time
            segmentSecondsRef.current = 0;
            lastCoachingTimeRef.current = now;

            let matchLevel: "good" | "developing" | "needs_work";
            if (score >= 75) {
              matchLevel = "good";
            } else if (score >= 50) {
              matchLevel = "developing";
            } else {
              matchLevel = "needs_work";
            }

            // Fire and forget; internal throttling prevents spam
            const validJoints = Object.keys(userMedianAngles).filter((k) => {
              const a = userMedianAngles[k];
              const c = userMedianConfidence[k] ?? 0;
              return typeof a === "number" && a !== 0 && c >= 0.3;
            });
            void sendCoachingRequest(activeSegmentId, userMedianAngles, userMedianConfidence, validJoints, matchLevel);
          }
        } else {
          // Not in a valid coaching state; reset dwell time
          segmentSecondsRef.current = 0;
        }
      }
    }, 1000); // Check every 1 second

    return () => clearInterval(compareInterval);
  }, [practiceMode, detectorReady, skeletonFrames, useTestVideo, activeSegmentId, coachingLoading, sendCoachingRequest]);

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

  const setRate = useCallback((rate: 1 | 0.75 | 0.5) => {
    setPlaybackRate(rate);
    if (videoRef.current) {
      videoRef.current.playbackRate = rate;
    }
  }, []);

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

      {/* Top bar */}
      <header className="flex items-center justify-between gap-4 px-4 py-3 border-b border-zinc-800 shrink-0">
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
            <button
              type="button"
              onClick={stopPractice}
              className="rounded-lg px-3 py-1.5 text-xs font-medium transition bg-zinc-700 text-white hover:bg-zinc-600"
            >
              Stop Practice
            </button>
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
          <QualityBadge score={overallScore} />
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
        </div>
      </header>

      {webcamError && (
        <div className="px-4 py-2 bg-red-500/10 border-b border-red-500/30 text-red-400 text-sm">
          {webcamError}
        </div>
      )}

      <div className="flex flex-1 min-h-0">
        {/* Main content: video panels */}
        <div className="flex-1 min-h-0 min-w-0 flex gap-2 p-2 sm:p-4 relative">
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
                className="absolute inset-0 w-full h-full object-contain"
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
            </div>
          </div>

          {/* Center: Coaching feedback panel (only in real practice mode) */}
          {practiceMode && !useTestVideo && (
            <div className="absolute left-1/2 -translate-x-1/2 top-4 z-30 flex flex-col items-center pointer-events-none">
              <div className="pointer-events-auto rounded-xl bg-black/70 border border-emerald-500/30 px-4 py-3 shadow-lg max-w-sm w-full transition-opacity duration-500">
                <div className="flex items-center gap-2 mb-1">
                  <div className={`h-2 w-2 rounded-full ${coachingLoading ? "bg-emerald-400 animate-pulse" : "bg-emerald-400/60"}`} />
                  <span className="text-xs font-semibold uppercase tracking-wide text-emerald-300">
                    Coaching
                  </span>
                </div>
                <p
                  className={`text-sm text-zinc-100 transition-opacity duration-500 ${
                    coachingNote ? "opacity-100" : "opacity-70"
                  }`}
                >
                  {coachingNote ??
                    "Hold this phrase for a few seconds while we watch your lines, then coaching tips will appear here."}
                </p>
                {worstJoint && coachingNote && (
                  <p className="mt-1 text-xs text-emerald-200/80">
                    Focus area: <span className="font-medium capitalize">{worstJoint}</span>
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Right: Webcam panel (only in webcam mode, not test mode) */}
          {practiceMode && !useTestVideo && (
            <div className="w-1/2 h-full flex items-center justify-center">
              <div
                ref={webcamContainerRef}
                className="w-full max-h-full aspect-video rounded-xl overflow-hidden bg-zinc-900 relative"
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
    </div>
  );
}
