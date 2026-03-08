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
  angle_summary?: AngleSummary;
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

function landmarkMap(landmarks: Landmark[]): Record<string, { x: number; y: number }> {
  const m: Record<string, { x: number; y: number }> = {};
  for (const lm of landmarks) {
    m[lm.name] = { x: lm.x, y: lm.y };
  }
  return m;
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

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const loopIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const videoContainerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

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

  // Draw skeleton overlay synced to video time
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

      // Calculate letterboxing offset: video content may not fill the entire element
      const videoWidth = video.videoWidth || 1;
      const videoHeight = video.videoHeight || 1;
      const containerWidth = rect.width;
      const containerHeight = rect.height;

      // object-fit: contain uses the smaller scale to fit the video
      const scaleX = containerWidth / videoWidth;
      const scaleY = containerHeight / videoHeight;
      const scale = Math.min(scaleX, scaleY);

      // Actual rendered video size
      const renderedWidth = videoWidth * scale;
      const renderedHeight = videoHeight * scale;

      // Offset to center the video content (letterboxing)
      const offsetX = (containerWidth - renderedWidth) / 2;
      const offsetY = (containerHeight - renderedHeight) / 2;

      // Transform normalized 0-1 landmark coords to canvas coords
      const toCanvas = (p: { x: number; y: number }) => ({
        x: offsetX + p.x * renderedWidth,
        y: offsetY + p.y * renderedHeight,
      });

      ctx.strokeStyle = "#00FF88";
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

      ctx.fillStyle = "#FFFFFF";
      const jointRadius = 4;
      for (const name of POSE_JOINTS) {
        const p = joints[name];
        if (!p) continue;
        const c = toCanvas(p);
        ctx.beginPath();
        ctx.arc(c.x, c.y, jointRadius, 0, Math.PI * 2);
        ctx.fill();
      }

      rafId = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(rafId);
  }, [showSkeleton, skeletonFrames]);

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
      {/* Top bar: back + quality badge + speed */}
      <header className="flex items-center justify-between gap-4 px-4 py-3 border-b border-zinc-800 shrink-0">
        <button
          type="button"
          onClick={() => router.push("/")}
          className="text-sm text-zinc-500 hover:text-white transition"
        >
          ← Home
        </button>
        <div className="flex items-center gap-4">
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

      <div className="flex flex-1 min-h-0">
        {/* Video — largest 16:9 box that fits, no cropping */}
        <div className="flex-1 min-h-0 min-w-0 flex items-center justify-center p-2 sm:p-4">
          <div className="w-full h-full flex items-center justify-center min-h-0 min-w-0">
            <div
              ref={videoContainerRef}
              className="w-full h-full max-w-full max-h-full aspect-video rounded-xl overflow-hidden bg-black relative"
            >
              <video
                ref={videoRef}
                src={`${API_URL}/video/${jobId}`}
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
            </div>
          </div>
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
