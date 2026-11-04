import type { Frame, Point } from "./types"

/** Below this visibility a landmark is treated as missing and filled from its neighbours. */
const MIN_VISIBILITY = 0.35

/**
 * Cleans the raw pose track: frames with no person, and landmarks the model
 * wasn't sure of, are filled by interpolating between good frames; then each
 * landmark is smoothed with a short centred moving average, which removes
 * frame-to-frame jitter without shifting events in time.
 * Returns null when fewer than half the frames found a person.
 */
export function cleanTrack(frames: Frame[], window = 5): Point[][] | null {
  const n = frames.length
  if (!n || frames.filter((f) => f.pts).length < n / 2) return null
  const count = frames.find((f) => f.pts)!.pts!.length
  const out: Point[][] = Array.from({ length: n }, () => [])

  for (let k = 0; k < count; k++) {
    const good = frames.map((f) => (f.pts && f.pts[k].v >= MIN_VISIBILITY ? f.pts[k] : null))
    const filled = interpolate(good)
    const half = Math.floor(window / 2)
    for (let i = 0; i < n; i++) {
      let sx = 0
      let sy = 0
      let m = 0
      for (let j = Math.max(0, i - half); j <= Math.min(n - 1, i + half); j++) {
        sx += filled[j].x
        sy += filled[j].y
        m++
      }
      out[i][k] = { x: sx / m, y: sy / m, v: good[i] ? good[i]!.v : 0 }
    }
  }
  return out
}

/** Fills nulls linearly between known values; leading and trailing gaps copy the nearest value. */
export function interpolate(values: (Point | null)[]): Point[] {
  const n = values.length
  const known = values.map((v, i) => (v ? i : -1)).filter((i) => i >= 0)
  if (!known.length) return values.map(() => ({ x: 0, y: 0, v: 0 }))
  const out: Point[] = new Array(n)
  for (let i = 0; i < n; i++) {
    if (values[i]) {
      out[i] = values[i]!
      continue
    }
    const after = known.find((k) => k > i)
    const before = [...known].reverse().find((k) => k < i)
    if (before === undefined) out[i] = values[after!]!
    else if (after === undefined) out[i] = values[before]!
    else {
      const a = values[before]!
      const b = values[after]!
      const f = (i - before) / (after - before)
      out[i] = { x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f, v: 0 }
    }
  }
  return out
}

export const mid = (a: Point, b: Point): Point => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, v: Math.min(a.v, b.v) })
export const dist = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y)

/** The angle at `b` in degrees, between the segments b→a and b→c. */
export function angleAt(a: Point, b: Point, c: Point): number {
  const v1 = Math.atan2(a.y - b.y, a.x - b.x)
  const v2 = Math.atan2(c.y - b.y, c.x - b.x)
  let d = Math.abs(v1 - v2) * (180 / Math.PI)
  if (d > 180) d = 360 - d
  return d
}

/** The angle of the segment from `low` up to `high`, measured from vertical, in degrees (0 = upright). */
export function tiltFromVertical(low: Point, high: Point): number {
  return Math.abs(Math.atan2(high.x - low.x, low.y - high.y)) * (180 / Math.PI)
}

export function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b)
  return s.length ? s[Math.floor(s.length / 2)] : 0
}
