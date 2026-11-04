import type { Fault, Metric } from "../analysis/metrics"
import { drillsFor, type Drill } from "./drills"

/**
 * Fix the foundations first: setup, then the body's movement, then the arms,
 * then rhythm and finish. A later fault is often caused by an earlier one.
 */
const PRIORITY: Fault[] = [
  "posture-upright",
  "posture-hunched",
  "knee-flex",
  "early-extension",
  "hip-sway",
  "head-sway",
  "head-height",
  "short-turn",
  "lead-arm-bent",
  "tempo-quick",
  "tempo-slow",
  "finish-balance",
]

export interface Plan {
  /** The one thing to work on first, or null when nothing needs work */
  first: { metric: Metric; drills: Drill[] } | null
  /** Other things to work on later, in priority order */
  later: Metric[]
  /** What's already good */
  strengths: Metric[]
}

export function makePlan(metrics: Metric[]): Plan {
  const rank = (m: Metric) => (m.fault ? PRIORITY.indexOf(m.fault) : 99)
  // "work" before "ok", then by foundation order
  const toFix = metrics
    .filter((m) => m.status !== "good" && m.fault)
    .sort((a, b) => (a.status === b.status ? rank(a) - rank(b) : a.status === "work" ? -1 : 1))
  const first = toFix[0]
  return {
    first: first ? { metric: first, drills: drillsFor(first.fault!).slice(0, 2) } : null,
    later: toFix.slice(1),
    strengths: metrics.filter((m) => m.status === "good"),
  }
}
