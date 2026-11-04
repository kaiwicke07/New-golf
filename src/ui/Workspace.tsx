import { useCallback, useEffect, useMemo, useRef, useState } from "react"

import { analyzeFrames, type Analysis } from "../analysis/analyze"
import type { Metric } from "../analysis/metrics"
import { PHASE_LABELS, type Hand, type Phases, type View } from "../analysis/types"
import { makePlan } from "../coach/plan"
import { saveSwing } from "../history"
import { trackPose } from "../pose/detect"
import { DrillCard } from "./DrillsPage"
import { PauseIcon, PlayIcon, StepBack, StepForward, ViewDiagram } from "./icons"
import { drawOverlay, frameAt, guidesFor, thumbnail, trailFor } from "./overlay"
import { SAMPLE_PARTS, type SamplePart } from "./samples"

export interface Source {
  url: string
  name: string
  /** The bundled sample: which part of the clip to analyse, and its camera view */
  sample?: SamplePart
}

type Stage = "setup" | "processing" | "review"

/** Longest stretch analysed at once; one swing takes 3–6 seconds in real time. */
const MAX_RANGE = 20
const PHASE_KEYS: (keyof Phases)[] = ["address", "top", "impact", "finish"]
const SPEEDS = [0.25, 0.5, 1]
const STATUS_TEXT = { good: "On target", ok: "Close", work: "Work on this" } as const

const fmt = (t: number) => `${t.toFixed(2)} s`

function seekTo(video: HTMLVideoElement, t: number): Promise<void> {
  return new Promise((resolve) => {
    if (Math.abs(video.currentTime - t) < 0.001) return resolve()
    video.addEventListener("seeked", () => resolve(), { once: true })
    video.currentTime = t
  })
}

/**
 * One video, three stages: choose the swing and the camera view, track the
 * pose, then review it like a launch monitor: replay on top, readouts below.
 */
export function Workspace({
  source,
  onReset,
  onShowDrills,
}: {
  source: Source
  onReset: () => void
  onShowDrills: (fault: string) => void
}) {
  const video = useRef<HTMLVideoElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const abort = useRef<AbortController | null>(null)
  const saved = useRef(false)

  const [duration, setDuration] = useState(0)
  const [range, setRange] = useState<[number, number]>([0, 0])
  const [view, setView] = useState<View>(source.sample?.view ?? "face-on")
  const [hand, setHand] = useState<Hand>("right")
  const [stage, setStage] = useState<Stage>("setup")
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(0.5)
  const [selected, setSelected] = useState<string | null>(null)

  const plan = useMemo(() => (analysis ? makePlan(analysis.metrics) : null), [analysis])
  const guides = useMemo(() => (analysis ? guidesFor(analysis) : {}), [analysis])

  // Redraw the overlay for the frame on screen
  const redraw = useCallback(() => {
    const v = video.current
    const c = canvas.current
    if (!v || !c) return
    if (!analysis) return drawOverlay(c, v, null)
    const i = frameAt(analysis.times, v.currentTime)
    drawOverlay(c, v, analysis.track[i], guides, trailFor(analysis, i))
  }, [analysis, guides])

  // While playing, draw every display frame and loop inside the chosen range
  useEffect(() => {
    if (!playing) return
    const frame = { id: 0 }
    const tick = () => {
      const v = video.current
      if (v) {
        if (v.currentTime >= range[1]) v.currentTime = range[0]
        setTime(v.currentTime)
        redraw()
      }
      frame.id = requestAnimationFrame(tick)
    }
    frame.id = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame.id)
  }, [playing, range, redraw])

  useEffect(() => {
    if (video.current) video.current.playbackRate = speed
  }, [speed])

  // Keep the canvas sized to the stage
  useEffect(() => {
    const c = canvas.current
    if (!c) return
    const ro = new ResizeObserver(() => redraw())
    ro.observe(c)
    return () => ro.disconnect()
  }, [redraw])

  const onLoaded = () => {
    const v = video.current!
    setDuration(v.duration)
    const start = source.sample ? source.sample.start : 0
    const end = source.sample ? source.sample.end : Math.min(v.duration, 15)
    setRange([start, end])
    v.currentTime = start
  }

  const analyze = async () => {
    const v = video.current!
    v.pause()
    setPlaying(false)
    setError(null)
    setStage("processing")
    setProgress(0)
    saved.current = false
    const controller = new AbortController()
    abort.current = controller
    try {
      const { frames } = await trackPose(
        v,
        range[0],
        range[1],
        (done, total, frame) => {
          setProgress(done / total)
          if (canvas.current) drawOverlay(canvas.current, v, frame.pts)
        },
        controller.signal,
      )
      // Dev only: lets tests read the raw track
      if (import.meta.env.DEV) (window as unknown as { __frames: unknown }).__frames = frames
      const result = analyzeFrames(frames, view, hand)
      if (!result.ok) {
        setError(result.reason)
        setStage("setup")
        return
      }
      setAnalysis(result.analysis)
      setStage("review")
      const impact = result.analysis.times[result.analysis.phases.impact]
      await seekTo(v, impact)
      setTime(impact)
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setError("Pose tracking couldn't start in this browser. Try the latest Chrome, Edge or Safari.")
      }
      setStage("setup")
    }
  }

  // Save to history once, with a picture of impact
  useEffect(() => {
    if (stage !== "review" || !analysis || saved.current || !video.current) return
    saved.current = true
    saveSwing({
      id: `${Date.now()}`,
      date: Date.now(),
      name: source.name,
      view: analysis.view,
      hand: analysis.hand,
      thumb: thumbnail(video.current),
      metrics: analysis.metrics.map(({ id, label, display, status }) => ({ id, label, display, status })),
    })
  }, [stage, analysis, source.name])

  useEffect(() => {
    if (stage === "review") redraw()
  }, [stage, time, redraw])

  const togglePlay = () => {
    const v = video.current
    if (!v) return
    if (v.paused) {
      if (v.currentTime >= range[1] - 0.01) v.currentTime = range[0]
      void v.play()
      setPlaying(true)
    } else {
      v.pause()
      setPlaying(false)
    }
  }

  const goTo = async (t: number) => {
    const v = video.current
    if (!v) return
    v.pause()
    setPlaying(false)
    await seekTo(v, Math.max(range[0], Math.min(range[1], t)))
    setTime(v.currentTime)
  }

  const step = (dir: 1 | -1) => {
    if (!analysis) return
    const i = frameAt(analysis.times, time)
    const next = Math.max(0, Math.min(analysis.times.length - 1, i + dir))
    void goTo(analysis.times[next])
  }
  const jumpPhase = (p: keyof Phases) => analysis && void goTo(analysis.times[analysis.phases[p]])

  // Keyboard: space plays, arrows step a frame, 1–4 jump to a phase
  useEffect(() => {
    if (stage !== "review") return
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLElement && e.target.closest("input, select, textarea")) return
      if (e.key === " ") {
        e.preventDefault()
        togglePlay()
      } else if (e.key === "ArrowLeft") step(-1)
      else if (e.key === "ArrowRight") step(1)
      else if (["1", "2", "3", "4"].includes(e.key)) jumpPhase(PHASE_KEYS[Number(e.key) - 1])
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  })

  const currentPhase = analysis
    ? PHASE_KEYS.find((p) => Math.abs(analysis.times[analysis.phases[p]] - time) < 0.02)
    : undefined
  const span = range[1] - range[0] || 1

  return (
    <div className="workspace">
      <div className="monitor">
        <div>
          <div className="stage">
            <video
              ref={video}
              src={source.url}
              playsInline
              muted
              preload="auto"
              controls={stage === "setup"}
              onLoadedMetadata={onLoaded}
              onSeeked={() => stage === "review" && redraw()}
              onPause={() => setPlaying(false)}
            />
            <canvas ref={canvas} aria-hidden="true" />
            {stage === "review" && (
              <span className={`stage-badge ${currentPhase ? "" : "is-time"}`}>
                {currentPhase ? PHASE_LABELS[currentPhase] : fmt(time - range[0])}
              </span>
            )}
          </div>

          {stage === "review" && analysis && (
            <>
              <div className="scrub">
                <div className="scrub-marks" aria-hidden="true">
                  {PHASE_KEYS.map((p) => (
                    <span
                      key={p}
                      className="scrub-mark"
                      style={{ left: `${((analysis.times[analysis.phases[p]] - range[0]) / span) * 100}%` }}
                    >
                      <span>{PHASE_LABELS[p]}</span>
                    </span>
                  ))}
                </div>
                <input
                  type="range"
                  min={range[0]}
                  max={range[1]}
                  step={0.001}
                  value={time}
                  aria-label="Position in the swing"
                  onChange={(e) => void goTo(Number(e.target.value))}
                />
              </div>
              <div className="transport">
                <button type="button" className="icon-btn" onClick={() => step(-1)} aria-label="Previous frame" title="Previous frame (←)">
                  <StepBack />
                </button>
                <button type="button" className="icon-btn is-primary" onClick={togglePlay} aria-label={playing ? "Pause" : "Play"} title="Play or pause (space)">
                  {playing ? <PauseIcon /> : <PlayIcon />}
                </button>
                <button type="button" className="icon-btn" onClick={() => step(1)} aria-label="Next frame" title="Next frame (→)">
                  <StepForward />
                </button>
                <div className="speed" role="group" aria-label="Playback speed">
                  {SPEEDS.map((s) => (
                    <button key={s} type="button" aria-pressed={speed === s} onClick={() => setSpeed(s)}>
                      {s}×
                    </button>
                  ))}
                </div>
                <span className="time num">
                  {fmt(time - range[0])} / {fmt(span)}
                </span>
              </div>
              <div className="phases" role="group" aria-label="Jump to a moment">
                {PHASE_KEYS.map((p, i) => (
                  <button key={p} type="button" className="phase" aria-pressed={currentPhase === p} onClick={() => jumpPhase(p)}>
                    <strong>{PHASE_LABELS[p]}</strong>
                    <span className="num">
                      {fmt(analysis.times[analysis.phases[p]] - range[0])}
                      <span className="key-hint"> · key {i + 1}</span>
                    </span>
                  </button>
                ))}
              </div>
            </>
          )}
        </div>

        <aside className="side">
          {stage === "setup" && (
            <Setup
              duration={duration}
              range={range}
              onRange={setRange}
              currentTime={() => video.current?.currentTime ?? 0}
              view={view}
              onView={setView}
              hand={hand}
              onHand={setHand}
              error={error}
              onAnalyze={analyze}
              onReset={onReset}
              sample={!!source.sample}
              onSamplePart={(part) => {
                setRange([part.start, part.end])
                setView(part.view)
                if (video.current) video.current.currentTime = part.start
              }}
            />
          )}

          {stage === "processing" && (
            <div className="side-block" role="status">
              <h2>Tracking</h2>
              <p className="big-percent num">{Math.round(progress * 100)}%</p>
              <div className="progress" aria-hidden="true">
                <div style={{ width: `${Math.round(progress * 100)}%` }} />
              </div>
              <p className="hint">Finding your joints in every frame, on this device. Nothing is uploaded.</p>
              <button type="button" className="btn btn-small" style={{ marginTop: 16 }} onClick={() => abort.current?.abort()}>
                Cancel
              </button>
            </div>
          )}

          {stage === "review" && analysis && (
            <>
              <div className="side-block">
                <span className="label">Session</span>
                <h2>{source.name}</h2>
                <p className="muted">
                  {analysis.view === "face-on" ? "Face-on" : "Down the line"} ·{" "}
                  {analysis.hand === "right" ? "Right-handed" : "Left-handed"}
                </p>
                <div className="tally">
                  <div className="is-good">
                    <b className="num">{analysis.metrics.filter((m) => m.status === "good").length}</b>
                    <span className="label">On target</span>
                  </div>
                  <div className="is-ok">
                    <b className="num">{analysis.metrics.filter((m) => m.status === "ok").length}</b>
                    <span className="label">Close</span>
                  </div>
                  <div className="is-work">
                    <b className="num">{analysis.metrics.filter((m) => m.status === "work").length}</b>
                    <span className="label">Work on</span>
                  </div>
                </div>
              </div>
              <div className="side-block">
                <span className="label">How to read it</span>
                <p className="hint">
                  Lime lines are your body; the glowing trail is the path of your hands. Dashed white marks show where
                  your head and hips started. Measured from one camera, so treat numbers as a guide.
                </p>
              </div>
              <div className="side-actions">
                <button
                  type="button"
                  className="btn btn-small"
                  onClick={() => {
                    setAnalysis(null)
                    setStage("setup")
                    if (canvas.current && video.current) drawOverlay(canvas.current, video.current, null)
                  }}
                >
                  Change settings
                </button>
                <button type="button" className="btn btn-primary btn-small" onClick={onReset}>
                  Analyse another swing
                </button>
              </div>
            </>
          )}
        </aside>
      </div>

      {stage === "review" && analysis && plan && (
        <>
          <section aria-labelledby="readouts-title">
            <h2 id="readouts-title" className="visually-hidden">
              Measurements
            </h2>
            <div className="readouts">
              {analysis.metrics.map((m) => (
                <Readout
                  key={m.id}
                  metric={m}
                  selected={selected === m.id}
                  onSelect={() => {
                    setSelected(m.id)
                    jumpPhase(m.phase)
                  }}
                />
              ))}
            </div>
          </section>

          <section className="coach" aria-labelledby="coach-title">
            <div className="coach-fix">
              <span className="label">{plan.first ? "Priority fix" : "Result"}</span>
              <h2 id="coach-title">{plan.first ? plan.first.metric.label : "Nothing major to fix"}</h2>
              {plan.first ? (
                <>
                  <p>
                    You: <b>{plan.first.metric.display}</b>. Aim: {plan.first.metric.target.toLowerCase()}.
                  </p>
                  <p>{plan.first.metric.why}</p>
                  <button type="button" className="btn btn-small" onClick={() => onShowDrills(plan.first!.metric.fault!)}>
                    All drills for this
                  </button>
                </>
              ) : (
                <p>Every check is on target or close. Keep your rhythm with the tempo drill and film again in a week.</p>
              )}
            </div>
            <div className="coach-drills">
              {(plan.first?.drills ?? []).map((d) => (
                <DrillCard key={d.id} drill={d} />
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  )
}

function Readout({ metric, selected, onSelect }: { metric: Metric; selected: boolean; onSelect: () => void }) {
  return (
    <button type="button" className={`readout is-${metric.status}`} aria-pressed={selected} onClick={onSelect}>
      <span className="label">{metric.label}</span>
      <span className="readout-value num">{metric.value}</span>
      <span className="readout-unit">{metric.unit}</span>
      <span className="readout-status">{STATUS_TEXT[metric.status]}</span>
      <span className="readout-target">
        Aim: {metric.target}
        {metric.estimate && " · estimate"}
      </span>
      {selected && <span className="readout-why">{metric.why}</span>}
    </button>
  )
}

function Setup({
  duration,
  range,
  onRange,
  currentTime,
  view,
  onView,
  hand,
  onHand,
  error,
  onAnalyze,
  onReset,
  sample,
  onSamplePart,
}: {
  duration: number
  range: [number, number]
  onRange: (r: [number, number]) => void
  currentTime: () => number
  view: View
  onView: (v: View) => void
  hand: Hand
  onHand: (h: Hand) => void
  error: string | null
  onAnalyze: () => void
  onReset: () => void
  sample: boolean
  onSamplePart: (part: SamplePart) => void
}) {
  const [start, end] = range
  const tooLong = end - start > MAX_RANGE
  return (
    <>
      {error && (
        <div className="side-block">
          <p className="error" role="alert">
            {error}
          </p>
        </div>
      )}
      <div className="side-block">
        <span className="label">Step 1</span>
        <h2>Pick one swing</h2>
        {sample ? (
          <>
            <p className="hint">This sample has four slow-motion swings from both angles. Pick one:</p>
            <div className="choice" role="group" aria-label="Sample swing">
              {SAMPLE_PARTS.map((part) => (
                <button
                  key={part.view}
                  type="button"
                  aria-pressed={view === part.view && Math.abs(start - part.start) < 0.01}
                  onClick={() => onSamplePart(part)}
                >
                  <strong>{part.view === "face-on" ? "Face-on" : "Down the line"}</strong>
                  <span className="num">
                    {part.start.toFixed(1)}–{part.end.toFixed(1)} s
                  </span>
                </button>
              ))}
            </div>
          </>
        ) : (
          <>
            <p className="hint">Pause just before address and set the start; pause after the finish and set the end.</p>
            <div className="trim">
              <div className="trim-row">
                <output className="num">{start.toFixed(2)}</output>
                <button type="button" className="btn btn-small" onClick={() => onRange([Math.min(currentTime(), end - 0.5), end])}>
                  Set start here
                </button>
              </div>
              <div className="trim-row">
                <output className="num">{end.toFixed(2)}</output>
                <button type="button" className="btn btn-small" onClick={() => onRange([start, Math.max(currentTime(), start + 0.5)])}>
                  Set end here
                </button>
              </div>
              <p className="hint num">
                {(end - start).toFixed(1)} s of {duration.toFixed(1)} s
                {tooLong && " · long for one swing; trim it for a faster, more accurate read"}
              </p>
            </div>
          </>
        )}
      </div>

      <div className="side-block">
        <span className="label">Step 2</span>
        <h2>Camera position</h2>
        <div className="choice" role="group" aria-label="Camera view">
          {(["face-on", "down-the-line"] as View[]).map((v) => (
            <button key={v} type="button" aria-pressed={view === v} onClick={() => onView(v)}>
              <ViewDiagram view={v} />
              <strong>{v === "face-on" ? "Face-on" : "Down the line"}</strong>
              <span>{v === "face-on" ? "Facing your chest" : "Behind your hands"}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="side-block">
        <span className="label">Step 3</span>
        <h2>Handedness</h2>
        <div className="choice" role="group" aria-label="Handedness">
          {(["right", "left"] as Hand[]).map((h) => (
            <button key={h} type="button" aria-pressed={hand === h} onClick={() => onHand(h)}>
              <strong>{h === "right" ? "Right" : "Left"}</strong>
              <span>{h === "right" ? "Target on your left" : "Target on your right"}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="side-actions">
        <button type="button" className="btn btn-primary" onClick={onAnalyze} disabled={end - start < 0.5}>
          Analyse swing
        </button>
        <button type="button" className="btn btn-ghost btn-small" onClick={onReset}>
          Choose a different video
        </button>
      </div>
    </>
  )
}
