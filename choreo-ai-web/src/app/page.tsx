"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Status = "idle" | "processing" | "complete" | "failed";

interface ChoreoData {
  video_id?: string;
  source_url?: string;
  quality_assessment?: {
    overall_score?: number;
    confidence_score?: number;
    continuity_score?: number;
    coverage_score?: number;
    flags?: string[];
  };
  segments?: unknown[];
}

interface StatusResponse {
  job_id: string;
  status: "pending" | "processing" | "complete" | "failed";
  error?: string;
  choreo_data?: ChoreoData;
  video_id?: string;
}

export default function Home() {
  const router = useRouter();
  const [url, setUrl] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [choreoData, setChoreoData] = useState<ChoreoData | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const pollStatus = useCallback(
    (id: string) => {
      pollRef.current = setInterval(async () => {
        try {
          const res = await fetch(`${API_URL}/status/${id}`);
          const data: StatusResponse = await res.json();
          if (data.status === "complete") {
            stopPolling();
            const dataChoreo = data.choreo_data ?? null;
            setChoreoData(dataChoreo);
            setStatus("complete");
            const vid = dataChoreo?.video_id ?? data.video_id;
            if (vid && id) {
              router.push(`/player/${vid}?jobId=${id}`);
            }
          } else if (data.status === "failed") {
            stopPolling();
            setError(data.error ?? "Analysis failed");
            setStatus("failed");
          }
        } catch (e) {
          stopPolling();
          setError(e instanceof Error ? e.message : "Request failed");
          setStatus("failed");
        }
      }, 3000);
    },
    [stopPolling]
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = url.trim();
    if (!trimmed) return;

    setError(null);
    setChoreoData(null);
    setStatus("processing");

    try {
      const res = await fetch(`${API_URL}/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ youtube_url: trimmed }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.detail ?? data.error ?? "Failed to start job");
        setStatus("failed");
        return;
      }
      const id = data.job_id as string;
      setJobId(id);
      pollStatus(id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Network error");
      setStatus("failed");
    }
  };

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  const handleRetry = () => {
    setStatus("idle");
    setError(null);
    setJobId(null);
    setChoreoData(null);
  };

  return (
    <div className="min-h-screen bg-[#0d0d0d] text-zinc-100 flex flex-col items-center justify-center px-4 font-sans">
      <div className="w-full max-w-lg mx-auto space-y-8">
        <header className="text-center">
          <h1 className="text-2xl font-semibold tracking-tight text-white">
            Choreo AI
          </h1>
          <p className="mt-1 text-sm text-zinc-500">
            Analyze dance choreography from a YouTube video
          </p>
        </header>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex flex-col sm:flex-row gap-3">
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Paste YouTube URL..."
              className="flex-1 h-12 px-4 rounded-xl bg-zinc-900 border border-zinc-800 text-white placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500 transition"
              disabled={status === "processing"}
            />
            <button
              type="submit"
              disabled={status === "processing" || !url.trim()}
              className="h-12 px-6 rounded-xl bg-emerald-600 text-white font-medium hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed transition shrink-0"
            >
              Analyze
            </button>
          </div>
        </form>

        {/* Status area */}
        <div className="min-h-[120px] rounded-xl bg-zinc-900/80 border border-zinc-800 p-6">
          {status === "idle" && (
            <p className="text-sm text-zinc-500 text-center">
              Enter a YouTube URL and click Analyze to start.
            </p>
          )}

          {status === "processing" && (
            <div className="flex flex-col items-center justify-center gap-4 py-2">
              <div className="size-10 border-2 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin" />
              <p className="text-sm text-zinc-400">Analyzing choreography...</p>
              <p className="text-xs text-zinc-600">This may take a few minutes.</p>
            </div>
          )}

          {status === "complete" && choreoData && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-emerald-400 text-sm font-medium">
                <span className="size-2 rounded-full bg-emerald-400" />
                Complete
              </div>
              <dl className="grid gap-3 text-sm">
                <div>
                  <dt className="text-zinc-500">Video ID</dt>
                  <dd className="font-mono text-white mt-0.5">
                    {choreoData.video_id ?? "—"}
                  </dd>
                </div>
                {choreoData.quality_assessment?.overall_score != null && (
                  <div>
                    <dt className="text-zinc-500">Quality score</dt>
                    <dd className="text-white mt-0.5">
                      {(choreoData.quality_assessment.overall_score * 100).toFixed(1)}%
                    </dd>
                  </div>
                )}
                <div>
                  <dt className="text-zinc-500">Segments</dt>
                  <dd className="text-white mt-0.5">
                    {Array.isArray(choreoData.segments)
                      ? choreoData.segments.length
                      : 0}{" "}
                    phrase(s)
                  </dd>
                </div>
              </dl>
            </div>
          )}

          {status === "failed" && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-red-400 text-sm font-medium">
                <span className="size-2 rounded-full bg-red-400" />
                Failed
              </div>
              <p className="text-sm text-zinc-300">{error ?? "Something went wrong."}</p>
              <button
                type="button"
                onClick={handleRetry}
                className="text-sm font-medium text-emerald-400 hover:text-emerald-300 transition"
              >
                Try again
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
