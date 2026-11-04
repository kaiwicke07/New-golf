import type { Analysis } from "../analysis/analyze"
import { mid } from "../analysis/series"
import { BONES, L, type Point } from "../analysis/types"

/** The nearest analysed frame to a playback time. */
export function frameAt(times: number[], t: number): number {
  let lo = 0
  let hi = times.length - 1
  while (lo < hi) {
    const m = (lo + hi) >> 1
    if (times[m] < t) lo = m + 1
    else hi = m
  }
  return lo > 0 && Math.abs(times[lo - 1] - t) < Math.abs(times[lo] - t) ? lo - 1 : lo
}

/** Reference marks from the address position, drawn under the moving skeleton. */
export interface Guides {
  /** Face-on: where the head started */
  head?: { x: number; y: number; r: number }
  /** Face-on: where the hips started, as a vertical line */
  hipLine?: number
  /** Down the line: the spine angle at address */
  spine?: { from: Point; to: Point }
}

export function guidesFor(a: Analysis): Guides {
  const p = a.track[a.phases.address]
  const hips = mid(p[L.leftHip], p[L.rightHip])
  const shoulders = mid(p[L.leftShoulder], p[L.rightShoulder])
  const torso = Math.hypot(shoulders.x - hips.x, shoulders.y - hips.y)
  if (a.view === "face-on") {
    return { head: { x: p[L.nose].x, y: p[L.nose].y, r: torso * 0.22 }, hipLine: hips.x }
  }
  // Extend the address spine line a little past the shoulders
  const to = { x: hips.x + (shoulders.x - hips.x) * 1.35, y: hips.y + (shoulders.y - hips.y) * 1.35, v: 1 }
  return { spine: { from: hips, to } }
}

const FLAG = "#f2b705"

/**
 * Draws the pose (and guides) over the video. Coordinates are video pixels;
 * the canvas is matched to the video's on-screen box (object-fit: contain).
 */
export function drawOverlay(canvas: HTMLCanvasElement, video: HTMLVideoElement, pts: Point[] | null, guides: Guides = {}) {
  const dpr = window.devicePixelRatio || 1
  const cw = canvas.clientWidth
  const ch = canvas.clientHeight
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
    canvas.width = Math.round(cw * dpr)
    canvas.height = Math.round(ch * dpr)
  }
  const ctx = canvas.getContext("2d")!
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, cw, ch)
  const vw = video.videoWidth
  const vh = video.videoHeight
  if (!vw || !vh) return
  const s = Math.min(cw / vw, ch / vh)
  const ox = (cw - vw * s) / 2
  const oy = (ch - vh * s) / 2
  const X = (x: number) => ox + x * s
  const Y = (y: number) => oy + y * s
  const line = Math.max(2, Math.min(4, 3 * s * (vh / 720)))

  // Guides first, dashed, in flag yellow
  ctx.save()
  ctx.setLineDash([8, 6])
  ctx.strokeStyle = FLAG
  ctx.lineWidth = 2
  if (guides.head) {
    ctx.beginPath()
    ctx.arc(X(guides.head.x), Y(guides.head.y), guides.head.r * s, 0, Math.PI * 2)
    ctx.stroke()
  }
  if (guides.hipLine !== undefined) {
    ctx.beginPath()
    ctx.moveTo(X(guides.hipLine), oy + 8)
    ctx.lineTo(X(guides.hipLine), oy + vh * s - 8)
    ctx.stroke()
  }
  if (guides.spine) {
    ctx.beginPath()
    ctx.moveTo(X(guides.spine.from.x), Y(guides.spine.from.y))
    ctx.lineTo(X(guides.spine.to.x), Y(guides.spine.to.y))
    ctx.stroke()
  }
  ctx.restore()

  if (!pts) return

  // The current spine, solid, so it can be compared with the address line
  if (guides.spine) {
    const hips = mid(pts[L.leftHip], pts[L.rightHip])
    const shoulders = mid(pts[L.leftShoulder], pts[L.rightShoulder])
    ctx.strokeStyle = FLAG
    ctx.lineWidth = line
    ctx.beginPath()
    ctx.moveTo(X(hips.x), Y(hips.y))
    ctx.lineTo(X(hips.x + (shoulders.x - hips.x) * 1.35), Y(hips.y + (shoulders.y - hips.y) * 1.35))
    ctx.stroke()
  }

  // Skeleton: a dark outline under white bones reads on any background
  ctx.lineCap = "round"
  for (const [width, colour] of [[line + 3, "rgb(13 21 17 / 0.55)"], [line, "#ffffff"]] as const) {
    ctx.strokeStyle = colour
    ctx.lineWidth = width
    ctx.beginPath()
    for (const [a, b] of BONES) {
      ctx.moveTo(X(pts[a].x), Y(pts[a].y))
      ctx.lineTo(X(pts[b].x), Y(pts[b].y))
    }
    ctx.stroke()
  }
  const joint = (i: number, r: number, colour: string) => {
    ctx.beginPath()
    ctx.arc(X(pts[i].x), Y(pts[i].y), r, 0, Math.PI * 2)
    ctx.fillStyle = colour
    ctx.fill()
    ctx.lineWidth = 2
    ctx.strokeStyle = "rgb(13 21 17 / 0.6)"
    ctx.stroke()
  }
  for (const i of [11, 12, 13, 14, 23, 24, 25, 26, 27, 28]) joint(i, line + 1, "#ffffff")
  joint(L.leftWrist, line + 2, FLAG)
  joint(L.rightWrist, line + 2, FLAG)
  joint(L.nose, line + 2, FLAG)
}

/** A small JPEG of the current video frame, for the history list. */
export function thumbnail(video: HTMLVideoElement, size = 160): string {
  const c = document.createElement("canvas")
  const s = size / Math.max(video.videoWidth, video.videoHeight)
  c.width = Math.round(video.videoWidth * s)
  c.height = Math.round(video.videoHeight * s)
  c.getContext("2d")!.drawImage(video, 0, 0, c.width, c.height)
  try {
    return c.toDataURL("image/jpeg", 0.72)
  } catch {
    return ""
  }
}
