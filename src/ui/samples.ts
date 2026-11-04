import type { View } from "../analysis/types"

export interface SamplePart {
  start: number
  end: number
  view: View
}

/** The sample clip holds four slow-motion swings; these two are offered, one per camera view. */
export const SAMPLE_PARTS: SamplePart[] = [
  { start: 17.6, end: 33.5, view: "face-on" },
  { start: 0.4, end: 16.8, view: "down-the-line" },
]
