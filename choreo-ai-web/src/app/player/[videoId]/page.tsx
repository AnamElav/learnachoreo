"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useSearchParams, useRouter } from "next/navigation";
import YouTube, { YouTubePlayer } from "react-youtube";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Segment {
  segment_id: number;
  start_time_ms: number;
  end_time_ms: number;
  beat_count: number;
}

interface ChoreoData {
  video_id?: string;
  quality_assessment?: { overall_score?: number };
  segments?: Segment[];
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

  const playerRef = useRef<YouTubePlayer | null>(null);
  const loopIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

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

  const stopLoop = useCallback(() => {
    if (loopIntervalRef.current) {
      clearInterval(loopIntervalRef.current);
      loopIntervalRef.current = null;
    }
    setActiveSegmentId(null);
  }, []);

  const playSegment = useCallback(
    (seg: Segment) => {
      const player = playerRef.current;
      if (!player?.seekTo) return;

      const startSec = seg.start_time_ms / 1000;
      const endSec = seg.end_time_ms / 1000;
      player.seekTo(startSec, true);
      player.playVideo();
      setActiveSegmentId(seg.segment_id);

      if (loopIntervalRef.current) clearInterval(loopIntervalRef.current);
      loopIntervalRef.current = setInterval(() => {
        const t = player.getCurrentTime?.();
        if (typeof t === "number" && t >= endSec - 0.1) {
          player.seekTo(startSec, true);
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

  const setRate = useCallback((rate: 1 | 0.75 | 0.5) => {
    setPlaybackRate(rate);
    if (playerRef.current?.setPlaybackRate) {
      playerRef.current.setPlaybackRate(rate);
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
        {/* Video — fills entire left panel */}
        <div className="flex-1 min-h-0 min-w-0 relative p-2 sm:p-4">
          <div className="absolute inset-2 sm:inset-4 rounded-xl overflow-hidden bg-black">
            <YouTube
              videoId={videoId}
              opts={{
                width: "100%",
                height: "100%",
                playerVars: {
                  autoplay: 0,
                  modestbranding: 1,
                },
              }}
              onReady={(e) => {
                playerRef.current = e.target;
                e.target.setPlaybackRate(playbackRate);
              }}
              onStateChange={(e) => {
                if (e.data === 0) stopLoop();
              }}
            />
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
