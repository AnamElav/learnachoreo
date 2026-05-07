"use client";

import type React from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  calculateJointAngles,
  drawSkeleton,
  makeFitNormalizer,
} from "@/lib/pose/skeletonPrimitives";
import {
  interpolatedAnglesAtTime,
  interpolatedJointsFromSkeleton,
  nearestUserKeyframe,
  swapLeftRightJointAngles,
  swapLeftRightPositions,
  computeWorstMoments,
  type PhraseMomentSnapshot,
  type UserPhraseKeyframe,
} from "@/lib/phraseReview/phraseReviewAnalysis";
import type { PhraseReviewSegmentMeta, SkeletonFrame } from "@/lib/phraseReview/types";

/** Half-width around each worst moment — full window 500 ms for yellow joint highlight */
const HIGHLIGHT_HALF_MS = 250;
/** Info card: opaque near center of a moment, fades out farther away */
const INFO_FULL_MS = 180;
const INFO_FADE_MS = 420;

function formatTimeLabel(ms: number) {
  const totalSec = Math.floor(ms / 1000);
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export type PhraseReviewProps = {
  apiUrl?: string;
  jobId: string | null;
  segment: PhraseReviewSegmentMeta;
  skeletonFrames: SkeletonFrame[];
  userKeyframes: UserPhraseKeyframe[];
  mirrorMapping: "direct" | "lr_swapped";
  practiceModeSide: "learn" | "dance";
  /** Main reference `<video>` — timeline and drawImage reuse this element */
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isReferenceMirrored: boolean;
  onGotIt: () => void;
};

function buildMomentReferenceAngleOverride(
  skeletonFrames: SkeletonFrame[],
  timeMs: number,
  mirrorMapping: "direct" | "lr_swapped"
): Record<string, number> {
  let ref = interpolatedAnglesAtTime(skeletonFrames as Parameters<typeof interpolatedAnglesAtTime>[0], timeMs);
  if (mirrorMapping === "lr_swapped") ref = swapLeftRightJointAngles(ref);
  return ref;
}

function focusWorstJointAtTime(timeMs: number, moments: PhraseMomentSnapshot[]): string | null {
  let bestKey: string | null = null;
  let bestDist = Infinity;
  for (const m of moments) {
    const d = Math.abs(timeMs - m.timestamp_ms);
    if (d <= HIGHLIGHT_HALF_MS && d < bestDist) {
      bestDist = d;
      bestKey = m.worstJointKey;
    }
  }
  return bestKey;
}

function dominantMomentNearTime(
  timeMs: number,
  moments: PhraseMomentSnapshot[]
): { moment: PhraseMomentSnapshot; opacity: number; noteIdx: number } | null {
  let bestD = Infinity;
  let idx = -1;

  moments.forEach((m, i) => {
    const d = Math.abs(timeMs - m.timestamp_ms);
    if (d < bestD) {
      bestD = d;
      idx = i;
    }
  });

  if (idx < 0 || idx >= moments.length) return null;
  const chosen = moments[idx]!;
  const d = Math.abs(timeMs - chosen.timestamp_ms);
  let opacity = 0;
  if (d <= INFO_FULL_MS) opacity = 1;
  else if (d < INFO_FADE_MS) {
    opacity = Math.max(0, 1 - (d - INFO_FULL_MS) / (INFO_FADE_MS - INFO_FULL_MS));
  }
  if (opacity < 1e-3) return null;

  return { moment: chosen, opacity, noteIdx: idx };
}

async function fetchCoachingSentence(opts: {
  apiUrl: string;
  jobId: string;
  segmentId: number;
  worstJointUpper: string;
  userAngles: Record<string, number>;
  refAnglesOverride: Record<string, number>;
  practiceModeSide: "learn" | "dance";
  signal: AbortSignal;
}): Promise<string | null> {
  const fakeConf: Record<string, number> = {};
  const keys = Object.keys(opts.userAngles);
  for (const k of keys) fakeConf[k] = 0.5;
  const valid = keys.filter((k) => opts.userAngles[k] !== undefined);
  const body: Record<string, unknown> = {
    segment_id: opts.segmentId,
    reference_angle_summary: opts.refAnglesOverride,
    user_angles: opts.userAngles,
    user_joint_confidence: fakeConf,
    valid_joints: valid.length ? valid : keys,
    match_level: "developing",
    skill_level: "beginner",
    style: "contemporary",
    practice_mode: opts.practiceModeSide,
    focus_joint: opts.worstJointUpper,
  };

  const res = await fetch(`${opts.apiUrl}/coaching`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: opts.signal,
  });
  if (!res.ok) return null;
  const data = (await res.json()) as { note?: string };
  return data.note ?? null;
}

function letterboxedVideoRects(
  containerW: number,
  containerH: number,
  videoW: number,
  videoH: number
) {
  const scaleX = containerW / videoW;
  const scaleY = containerH / videoH;
  const scale = Math.min(scaleX, scaleY);
  const renderedWidth = videoW * scale;
  const renderedHeight = videoH * scale;
  const offsetX = (containerW - renderedWidth) / 2;
  const offsetY = (containerH - renderedHeight) / 2;
  return { renderedWidth, renderedHeight, offsetX, offsetY };
}

export function PhraseReview({
  apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  jobId,
  segment,
  skeletonFrames,
  userKeyframes,
  mirrorMapping,
  practiceModeSide,
  videoRef,
  isReferenceMirrored,
  onGotIt,
}: PhraseReviewProps) {
  const phraseStartMs = segment.start_time_ms;
  const phraseEndMs = segment.end_time_ms;
  const phraseDurMs = Math.max(1, phraseEndMs - phraseStartMs);

  const moments = useMemo(() => {
    return computeWorstMoments({
      userKeyframes,
      skeletonFrames,
      phraseStartMs,
      phraseEndMs,
      mirrorMapping,
      topN: 3,
    });
  }, [userKeyframes, skeletonFrames, phraseStartMs, phraseEndMs, mirrorMapping]);

  const [notes, setNotes] = useState<(string | null)[]>([]);
  const [loadingNotes, setLoadingNotes] = useState(true);

  useEffect(() => {
    let cancelled = false;

    const jid = jobId;
    if (moments.length === 0) {
      const t = window.setTimeout(() => {
        if (cancelled) return;
        setNotes([]);
        setLoadingNotes(false);
      }, 0);
      return () => {
        cancelled = true;
        window.clearTimeout(t);
      };
    }

    if (!jid) {
      const t = window.setTimeout(() => {
        if (cancelled) return;
        setNotes(
          moments.map(
            () =>
              "Start analysis from home with a video job to enable coaching notes."
          )
        );
        setLoadingNotes(false);
      }, 0);
      return () => {
        cancelled = true;
        window.clearTimeout(t);
      };
    }

    const jobIdForApi: string = jid;
    const ac = new AbortController();

    async function load() {
      setLoadingNotes(true);
      const rows: string[] = [];
      for (let i = 0; i < moments.length; i++) {
        const m = moments[i]!;
        const userAngles = calculateJointAngles(m.userJoints);
        const refOverride = buildMomentReferenceAngleOverride(skeletonFrames, m.timestamp_ms, mirrorMapping);

        const note = await fetchCoachingSentence({
          apiUrl,
          jobId: jobIdForApi,
          segmentId: segment.segment_id,
          worstJointUpper: m.worstJointKey,
          userAngles,
          refAnglesOverride: refOverride,
          practiceModeSide,
          signal: ac.signal,
        });
        rows.push(note ?? `Try aligning your ${m.worstJointLabel.toLowerCase()} closer to the reference at this beat.`);
      }
      setNotes(rows);
      setLoadingNotes(false);
    }

    void load();

    return () => ac.abort();
  }, [apiUrl, jobId, moments, segment.segment_id, skeletonFrames, mirrorMapping, practiceModeSide]);

  const leftWrapRef = useRef<HTMLDivElement>(null);
  const rightWrapRef = useRef<HTMLDivElement>(null);
  const leftCanvasRef = useRef<HTMLCanvasElement>(null);
  const rightCanvasRef = useRef<HTMLCanvasElement>(null);
  const timelineBarRef = useRef<HTMLDivElement>(null);

  const momentsRef = useRef(moments);
  const skeletonRef = useRef(skeletonFrames);
  const userKfsRef = useRef(userKeyframes);

  const mirrorRef = useRef(mirrorMapping);
  const mirroredVidRef = useRef(isReferenceMirrored);

  const phraseStartRef = useRef(phraseStartMs);
  const phraseEndRef = useRef(phraseEndMs);

  useEffect(() => {
    momentsRef.current = moments;
    skeletonRef.current = skeletonFrames;
    userKfsRef.current = userKeyframes;
    mirrorRef.current = mirrorMapping;
    mirroredVidRef.current = isReferenceMirrored;
    phraseStartRef.current = phraseStartMs;
    phraseEndRef.current = phraseEndMs;
  }, [
    moments,
    skeletonFrames,
    userKeyframes,
    mirrorMapping,
    isReferenceMirrored,
    phraseStartMs,
    phraseEndMs,
  ]);

  const playingRef = useRef(false);

  /** Drives timeline + info opacity from the real media clock */
  const [uiTimeMs, setUiTimeMs] = useState(phraseStartMs);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);

  /** Sync phrase review with main video timeline (reuse chunk-drill semantics) */
  useEffect(() => {
    const v = videoRef.current;
    if (!v || moments.length === 0) return;

    const syncUi = () => {
      let t = v.currentTime * 1000;
      const ps = phraseStartRef.current;
      const pe = phraseEndRef.current;
      if (t < ps) t = ps;
      if (t > pe) t = pe;
      setUiTimeMs(t);
    };

    const bootstrap = () => {
      v.currentTime = phraseStartMs / 1000;
      v.pause();
      setPlaying(false);
      syncUi();
    };
    bootstrap();

    v.addEventListener("timeupdate", syncUi);
    v.addEventListener("seeked", syncUi);

    return () => {
      v.removeEventListener("timeupdate", syncUi);
      v.removeEventListener("seeked", syncUi);
    };
  }, [videoRef, phraseStartMs, phraseEndMs, moments]);

  /** Loop reference phrase range while playing (attempt replay pattern) */
  useEffect(() => {
    const v = videoRef.current;
    if (!playing || moments.length === 0 || !v) return;

    const id = window.setInterval(() => {
      const el = videoRef.current;
      if (!el || !playingRef.current) return;
      const endSec = phraseEndRef.current / 1000;
      if (el.currentTime >= endSec - 0.08) {
        el.currentTime = phraseStartRef.current / 1000;
      }
    }, 200);

    return () => clearInterval(id);
  }, [playing, moments.length, videoRef]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (!playing || moments.length === 0) return;
    void v.play().catch(() => {});
  }, [playing, moments.length, videoRef]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (!playing) v.pause();
  }, [playing, videoRef]);

  const seekPhraseTimeMs = useCallback(
    (ms: number) => {
      const v = videoRef.current;
      if (!v) return;
      const clamped = Math.min(Math.max(ms, phraseStartMs), phraseEndMs);
      v.currentTime = clamped / 1000;
      setUiTimeMs(clamped);
    },
    [phraseStartMs, phraseEndMs, videoRef]
  );

  const togglePlay = useCallback(() => {
    setPlaying((p) => !p);
  }, []);

  const onTimelineSeek = useCallback(
    (clientX: number, barEl: HTMLDivElement) => {
      const rect = barEl.getBoundingClientRect();
      const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      const ms = phraseStartMs + ratio * phraseDurMs;
      seekPhraseTimeMs(ms);
    },
    [phraseStartMs, phraseDurMs, seekPhraseTimeMs]
  );

  const infoCardState = useMemo(
    () => dominantMomentNearTime(uiTimeMs, moments),
    [uiTimeMs, moments]
  );

  /** RAF: reference video composite + skeleton; user skeleton (chunk replay sync via nearestUserKeyframe) */
  useEffect(() => {
    if (moments.length === 0) return;

    let rafId = 0;

    function tick() {
      const v = videoRef.current;
      const leftCv = leftCanvasRef.current;
      const rightCv = rightCanvasRef.current;
      const leftWrap = leftWrapRef.current;
      const rightWrap = rightWrapRef.current;

      const mom = momentsRef.current;
      const sk = skeletonRef.current;
      const uvs = userKfsRef.current;
      const map = mirrorRef.current;

      let timeMs = (v?.currentTime ?? phraseStartRef.current / 1000) * 1000;
      const ps = phraseStartRef.current;
      const pe = phraseEndRef.current;
      timeMs = Math.min(Math.max(timeMs, ps), pe);

      const focusJoint = focusWorstJointAtTime(timeMs, mom);

      if (leftWrap && leftCv && v && v.videoWidth > 0 && v.videoHeight > 0) {
        const rect = leftWrap.getBoundingClientRect();
        const dpr = Math.min(2, typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1);
        const cw = rect.width;
        const ch = rect.height;

        leftCv.width = cw * dpr;
        leftCv.height = ch * dpr;
        leftCv.style.width = `${cw}px`;
        leftCv.style.height = `${ch}px`;

        const ctx = leftCv.getContext("2d");
        if (ctx) {
          ctx.setTransform(1, 0, 0, 1, 0, 0);
          ctx.scale(dpr, dpr);
          ctx.fillStyle = "#0f0f12";
          ctx.fillRect(0, 0, cw, ch);

          const { renderedWidth, renderedHeight, offsetX, offsetY } = letterboxedVideoRects(
            cw,
            ch,
            v.videoWidth,
            v.videoHeight
          );
          ctx.save();
          if (mirroredVidRef.current) {
            ctx.translate(offsetX + renderedWidth, offsetY);
            ctx.scale(-1, 1);
            ctx.drawImage(v, 0, 0, renderedWidth, renderedHeight);
          } else {
            ctx.translate(offsetX, offsetY);
            ctx.drawImage(v, 0, 0, renderedWidth, renderedHeight);
          }
          ctx.restore();

          const refJointsRaw = interpolatedJointsFromSkeleton(
            sk as Parameters<typeof interpolatedJointsFromSkeleton>[0],
            timeMs
          );
          const refDraw = map === "lr_swapped" ? swapLeftRightPositions(refJointsRaw) : refJointsRaw;
          const toCanvas = (p: { x: number; y: number }) => ({
            x: offsetX + (mirroredVidRef.current ? 1 - p.x : p.x) * renderedWidth,
            y: offsetY + p.y * renderedHeight,
          });

          drawSkeleton(ctx, refDraw, toCanvas, "#00FF88", "#FFFFFF", {
            focusJoint,
            highlightColor: "#FBBF24",
            baseLineWidth: 2.5,
            jointRadius: 4,
          });
        }
      }

      if (rightWrap && rightCv) {
        const rect = rightWrap.getBoundingClientRect();
        const dpr = Math.min(2, typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1);
        const cw = rect.width;
        const ch = rect.height;
        rightCv.width = cw * dpr;
        rightCv.height = ch * dpr;
        rightCv.style.width = `${cw}px`;
        rightCv.style.height = `${ch}px`;

        const ctx = rightCv.getContext("2d");
        if (ctx) {
          ctx.setTransform(1, 0, 0, 1, 0, 0);
          ctx.scale(dpr, dpr);
          ctx.fillStyle = "#0f0f12";
          ctx.fillRect(0, 0, cw, ch);

          const kf = nearestUserKeyframe(uvs, timeMs);
          const jointsRaw = kf?.joints ?? null;
          const joints =
            jointsRaw && mirroredVidRef.current
              ? Object.fromEntries(
                  Object.entries(jointsRaw).map(([k, v]) => [k, { x: 1 - v.x, y: v.y }])
                )
              : jointsRaw;
          if (joints && Object.keys(joints).length > 0) {
            const toSkin = makeFitNormalizer(joints, cw, ch, 12);
            drawSkeleton(ctx, joints, toSkin, "#FF6B6B", "#FFFFFF", {
              focusJoint,
              highlightColor: "#FBBF24",
              baseLineWidth: 2.5,
              jointRadius: 4,
            });
          }
        }
      }

      rafId = requestAnimationFrame(tick);
    }

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [moments.length, videoRef]);

  const playheadFrac = Math.min(
    1,
    Math.max(0, (uiTimeMs - phraseStartMs) / phraseDurMs)
  );

  return (
    <div className="fixed inset-0 z-[70] flex flex-col bg-black/88 backdrop-blur-sm">
      <div className="shrink-0 border-b border-zinc-800 px-4 py-3 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-white">Phrase playback review</h2>
          <p className="text-xs text-zinc-400">
            Synced reference video and your recorded skeleton. Yellow marks the toughest joint windows; dots jump to highlighted moments.
          </p>
        </div>
        <button
          type="button"
          onClick={onGotIt}
          className="rounded-xl bg-emerald-500 hover:bg-emerald-400 px-6 py-2.5 text-sm font-semibold text-white shadow-lg"
        >
          Got it
        </button>
      </div>

      <div className="flex-1 flex flex-col min-h-0 p-4 gap-4">
        {moments.length === 0 ? (
          <p className="text-zinc-400 text-center mt-16 max-w-md mx-auto">
            We didn&apos;t capture enough pose snapshots for this phrase to build a review — stay in frame and try again.
          </p>
        ) : (
          <>
            <div className="relative flex-1 flex flex-col lg:flex-row gap-4 min-h-0 items-stretch">
              <div
                ref={leftWrapRef}
                className="relative flex-1 min-h-[200px] rounded-xl overflow-hidden border border-emerald-500/30 bg-zinc-950"
              >
                <div className="absolute top-2 left-2 z-10 rounded bg-black/60 px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-emerald-400">
                  Reference video
                </div>
                <canvas ref={leftCanvasRef} className="absolute inset-0 w-full h-full" />
              </div>

              <div
                ref={rightWrapRef}
                className="relative flex-1 min-h-[200px] rounded-xl overflow-hidden border border-red-500/30 bg-zinc-950"
              >
                <div className="absolute top-2 left-2 z-10 rounded bg-black/60 px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-red-400">
                  You (replay)
                </div>
                <canvas ref={rightCanvasRef} className="absolute inset-0 w-full h-full" />
              </div>

              {/* Floating info card — opacity from dominant nearest moment */}
              {infoCardState && infoCardState.opacity > 0.02 ? (
                <div
                  className="pointer-events-none absolute bottom-14 left-1/2 z-20 w-[min(92vw,400px)] -translate-x-1/2 transition-opacity duration-150"
                  style={{ opacity: infoCardState.opacity }}
                >
                  <div className="rounded-xl border border-amber-500/35 bg-black/90 px-4 py-3 shadow-xl backdrop-blur-sm">
                    <p className="text-xs uppercase tracking-wide text-amber-300 font-semibold">
                      Moment · {formatTimeLabel(infoCardState.moment.timestamp_ms)}
                    </p>
                    <p className="text-sm text-white mt-1">
                      <span className="capitalize">{infoCardState.moment.worstJointLabel}</span>: your{" "}
                      <span className="font-mono tabular-nums">{infoCardState.moment.userDeg}°</span> vs reference{" "}
                      <span className="font-mono tabular-nums">{infoCardState.moment.refDeg}°</span>
                    </p>
                    <p className="text-xs text-zinc-400 mt-2 leading-snug">
                      {loadingNotes ? (
                        <span className="italic text-zinc-500">Fetching coaching tip…</span>
                      ) : (
                        notes[infoCardState.noteIdx] ?? "—"
                      )}
                    </p>
                  </div>
                </div>
              ) : null}
            </div>

            <div className="shrink-0 flex flex-wrap items-center gap-3 pb-2">
              <button
                type="button"
                aria-label={playing ? "Pause" : "Play"}
                onClick={togglePlay}
                className="flex h-10 w-10 items-center justify-center rounded-full border border-zinc-600 bg-zinc-900 text-zinc-100 hover:bg-zinc-800 shadow"
              >
                {playing ? (
                  <span className="text-lg leading-none">⏸</span>
                ) : (
                  <span className="text-lg leading-none pl-0.5">▶</span>
                )}
              </button>
              <div className="text-xs tabular-nums text-zinc-400">
                {formatTimeLabel(Math.round(uiTimeMs - phraseStartMs))}{" "}
                <span className="text-zinc-600">/</span> {formatTimeLabel(phraseDurMs)}
              </div>
            </div>

            {/* Timeline */}
            <div className="shrink-0 relative px-1 pb-2">
              <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1">Phrase timeline</div>
              <div
                ref={timelineBarRef}
                role="slider"
                tabIndex={0}
                aria-valuemin={0}
                aria-valuemax={phraseDurMs}
                aria-valuenow={uiTimeMs - phraseStartMs}
                className="relative h-3 cursor-pointer rounded-full bg-zinc-800 border border-zinc-700 shadow-inner outline-none ring-offset-2 ring-offset-black focus-visible:ring-2 focus-visible:ring-amber-400/70"
                onMouseDown={(e) => {
                  const bar = e.currentTarget;
                  onTimelineSeek(e.clientX, bar);
                  const mm = (ev: MouseEvent) => onTimelineSeek(ev.clientX, bar);
                  const mu = () => {
                    window.removeEventListener("mousemove", mm);
                    window.removeEventListener("mouseup", mu);
                  };
                  window.addEventListener("mousemove", mm);
                  window.addEventListener("mouseup", mu);
                }}
                onTouchStart={(e) => {
                  const bar = e.currentTarget;
                  const x = e.touches[0]?.clientX;
                  if (x != null) onTimelineSeek(x, bar);
                }}
                onTouchMove={(e) => {
                  const bar = timelineBarRef.current;
                  const x = e.touches[0]?.clientX;
                  if (bar && x != null) onTimelineSeek(x, bar);
                }}
              >
                {/* Moment markers */}
                {moments.map((m, i) => {
                  const frac = (m.timestamp_ms - phraseStartMs) / phraseDurMs;
                  return (
                    <button
                      key={`${m.timestamp_ms}-${i}`}
                      type="button"
                      aria-label={`Jump to highlight at ${formatTimeLabel(m.timestamp_ms)}`}
                      className="absolute top-1/2 z-10 -translate-x-1/2 -translate-y-1/2 rounded-full bg-amber-400 hover:bg-amber-300 hover:scale-125 w-3 h-3 border border-amber-950 shadow-md transition-transform"
                      style={{ left: `${frac * 100}%` }}
                      onMouseDown={(e) => {
                        e.stopPropagation();
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        seekPhraseTimeMs(m.timestamp_ms);
                      }}
                    />
                  );
                })}

                {/* Playhead */}
                <div
                  className="pointer-events-none absolute top-[-4px] z-[5] w-px h-[calc(100%+8px)] bg-white/90 rounded shadow"
                  style={{ left: `${playheadFrac * 100}%`, transform: "translateX(-50%)" }}
                />
              </div>
              <p className="text-[10px] text-zinc-600 mt-1">Click the bar or a yellow dot to seek both sides together.</p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
