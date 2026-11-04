import { dist, median, mid } from "./series"
import { L, type Phases, type Point } from "./types"

/**
 * The hands count as still below this share of the clip's top hand speed.
 * Relative, so slow-motion clips work the same as real-time ones.
 */
const STILL = 0.08

/** The hands: halfway between the wrists, since both hold the club. */
export const handsAt = (pts: Point[]) => mid(pts[L.leftWrist], pts[L.rightWrist])

/** Shoulder centre to hip centre: the body's own ruler, so speeds don't depend on how far away the camera was. */
export function torsoLength(track: Point[][]): number {
  return median(
    track.map((p) => dist(mid(p[L.leftShoulder], p[L.rightShoulder]), mid(p[L.leftHip], p[L.rightHip]))),
  )
}

export type PhaseResult = { ok: true; phases: Phases } | { ok: false; reason: string }

/**
 * Finds the four moments of the swing from how the hands move.
 *
 * - **Impact** is near the fastest hand speed of the clip, the bottom of the downswing:
 *   the frame around that peak where the hands are lowest.
 * - **Top** is the highest the hands get before impact.
 * - **Address** is the last still moment before the hands start back.
 * - **Finish** is the highest the hands get after impact, once they settle.
 */
export function findPhases(track: Point[][], times: number[]): PhaseResult {
  const n = track.length
  if (n < 12) return { ok: false, reason: "The clip is too short to contain a swing." }
  const scale = torsoLength(track) || 1
  const hands = track.map(handsAt)

  const speed = hands.map((_, i) => {
    const a = hands[Math.max(0, i - 1)]
    const b = hands[Math.min(n - 1, i + 1)]
    const dt = times[Math.min(n - 1, i + 1)] - times[Math.max(0, i - 1)] || 1
    return dist(a, b) / scale / dt
  })

  const peakSpeed = Math.max(...speed)
  const peak = speed.indexOf(peakSpeed)
  const still = peakSpeed * STILL
  if (peak < 3) return { ok: false, reason: "The swing starts too close to the beginning of the clip. Start recording a moment earlier." }

  // How high the hands go, and where they are at the bottom, so stillness can
  // be told apart: the golfer is also still at the top of a slow backswing
  let highest = Infinity
  for (let i = 0; i <= peak; i++) highest = Math.min(highest, hands[i].y)
  let lowest = -Infinity
  for (let i = Math.max(0, peak - 3); i <= Math.min(n - 1, peak + 6); i++) lowest = Math.max(lowest, hands[i].y)
  const handsLow = (i: number) => hands[i].y > highest + 0.7 * (lowest - highest)

  // Address: walking back from the fastest moment, the first spot where the
  // hands have been still, and low, for a few frames. Searching back from the
  // peak (not forward from the start) ignores a previous swing earlier in the clip.
  let address = -1
  let stillRun = 0
  for (let i = peak; i >= 0; i--) {
    stillRun = speed[i] < still && handsLow(i) ? stillRun + 1 : 0
    if (stillRun >= 3) {
      address = i + 2
      break
    }
  }
  // A clip that starts mid-takeaway: use the first low frame
  if (address < 0) address = handsLow(0) ? 0 : -1
  if (address < 0) return { ok: false, reason: "Couldn't find the address position. Start recording while the golfer is still, before the swing." }

  // Top: hands highest (smallest y) between address and the speed peak
  let top = address
  for (let i = address; i < peak; i++) if (hands[i].y < hands[top].y) top = i
  if (top - address < 2) return { ok: false, reason: "Couldn't find the backswing. Make sure the whole swing, from address, is in the clip." }

  // Impact: lowest hands between the top and a moment after the speed peak
  const window = Math.max(3, Math.round(0.12 / ((times[n - 1] - times[0]) / (n - 1) || 1 / 30)))
  let impact = top + 1
  for (let i = top + 1; i <= Math.min(n - 1, peak + window); i++) if (hands[i].y > hands[impact].y) impact = i
  if (impact <= top) return { ok: false, reason: "Couldn't find impact. Check that the golfer is fully in the frame." }

  // Finish: highest hands after impact
  let finish = impact + 1
  if (finish >= n) return { ok: false, reason: "The clip ends at impact. Keep recording until the finish." }
  for (let i = impact + 1; i < n; i++) if (hands[i].y < hands[finish].y) finish = i

  // Sanity: the hands must actually rise in the backswing, by at least half a torso
  if (hands[address].y - hands[top].y < scale * 0.5) {
    return { ok: false, reason: "The hands barely move. This doesn't look like a full swing." }
  }
  return { ok: true, phases: { address, top, impact, finish } }
}
