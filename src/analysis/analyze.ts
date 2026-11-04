import { measure, type Metric } from "./metrics"
import { findPhases } from "./phases"
import { cleanTrack } from "./series"
import type { Frame, Hand, Phases, Point, View } from "./types"

export interface Analysis {
  view: View
  hand: Hand
  times: number[]
  track: Point[][]
  phases: Phases
  metrics: Metric[]
}

export type AnalysisResult = { ok: true; analysis: Analysis } | { ok: false; reason: string }

/** From raw pose frames to phases and measurements. Pure, so it's tested without a browser. */
export function analyzeFrames(frames: Frame[], view: View, hand: Hand): AnalysisResult {
  const track = cleanTrack(frames)
  if (!track) {
    return {
      ok: false,
      reason: "Couldn't find a golfer in most of the clip. Make sure your whole body is in the frame and well lit.",
    }
  }
  const times = frames.map((f) => f.t)
  const found = findPhases(track, times)
  if (!found.ok) return found
  const metrics = measure({ track, times, phases: found.phases, view, hand })
  return { ok: true, analysis: { view, hand, times, track, phases: found.phases, metrics } }
}
