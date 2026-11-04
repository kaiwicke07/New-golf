import { describe, expect, it } from "vitest"

import { analyzeFrames } from "../src/analysis/analyze"
import { findPhases } from "../src/analysis/phases"
import { angleAt, cleanTrack, interpolate, tiltFromVertical } from "../src/analysis/series"
import { L, type Frame, type Point } from "../src/analysis/types"
import { makePlan } from "../src/coach/plan"
import { DRILLS } from "../src/coach/drills"

const FPS = 30

/** A standing golfer facing the camera; only the hands (and optionally the head) move. */
function body(hands: { x: number; y: number }, head = { x: 500, y: 200 }): Point[] {
  const pts: Point[] = Array.from({ length: 33 }, () => ({ x: 500, y: 500, v: 1 }))
  const set = (i: number, x: number, y: number) => (pts[i] = { x, y, v: 1 })
  set(L.nose, head.x, head.y)
  set(L.leftShoulder, 560, 300)
  set(L.rightShoulder, 440, 300)
  set(L.leftElbow, 545, 380)
  set(L.rightElbow, 455, 380)
  set(L.leftHip, 540, 520)
  set(L.rightHip, 460, 520)
  set(L.leftKnee, 545, 650)
  set(L.rightKnee, 455, 650)
  set(L.leftAnkle, 550, 780)
  set(L.rightAnkle, 450, 780)
  set(L.leftWrist, hands.x + 8, hands.y)
  set(L.rightWrist, hands.x - 8, hands.y)
  return pts
}

/** Still at address, a 0.9 s backswing, a 0.3 s downswing (3 : 1), a finish, then still. */
function syntheticSwing(): Frame[] {
  const ys: number[] = []
  for (let i = 0; i < 15; i++) ys.push(600)
  for (let i = 1; i <= 27; i++) ys.push(600 - (400 * i) / 27) // backswing up to y=200
  for (let i = 1; i <= 9; i++) ys.push(200 + (420 * i) / 9) // downswing to y=620
  for (let i = 1; i <= 12; i++) ys.push(620 - (470 * i) / 12) // follow-through to y=150
  for (let i = 0; i < 15; i++) ys.push(150)
  return ys.map((y, i) => ({ t: i / FPS, pts: body({ x: 500, y }) }))
}

describe("series", () => {
  it("interpolates gaps and copies ends", () => {
    const p = (x: number) => ({ x, y: 0, v: 1 })
    const out = interpolate([null, p(0), null, p(10), null])
    expect(out.map((q) => q.x)).toEqual([0, 0, 5, 10, 10])
  })

  it("measures angles", () => {
    expect(angleAt({ x: 0, y: 0, v: 1 }, { x: 1, y: 0, v: 1 }, { x: 1, y: 1, v: 1 })).toBeCloseTo(90)
    expect(angleAt({ x: 0, y: 0, v: 1 }, { x: 1, y: 0, v: 1 }, { x: 2, y: 0, v: 1 })).toBeCloseTo(180)
    expect(tiltFromVertical({ x: 0, y: 10, v: 1 }, { x: 0, y: 0, v: 1 })).toBeCloseTo(0)
    expect(tiltFromVertical({ x: 0, y: 10, v: 1 }, { x: 10, y: 0, v: 1 })).toBeCloseTo(45)
  })

  it("gives up when most frames have no person", () => {
    const frames: Frame[] = [{ t: 0, pts: null }, { t: 1, pts: null }, { t: 2, pts: body({ x: 0, y: 0 }) }]
    expect(cleanTrack(frames)).toBeNull()
  })
})

describe("findPhases", () => {
  it("finds address, top, impact and finish in order", () => {
    const frames = syntheticSwing()
    const track = cleanTrack(frames)!
    const result = findPhases(track, frames.map((f) => f.t))
    expect(result.ok).toBe(true)
    if (!result.ok) return
    const { address, top, impact, finish } = result.phases
    expect(address).toBeGreaterThanOrEqual(12)
    expect(address).toBeLessThanOrEqual(17)
    expect(top).toBeGreaterThanOrEqual(40)
    expect(top).toBeLessThanOrEqual(43)
    expect(impact).toBeGreaterThanOrEqual(49)
    expect(impact).toBeLessThanOrEqual(53)
    expect(finish).toBeGreaterThan(impact)
  })

  it("explains a clip with no swing in it", () => {
    const frames = Array.from({ length: 40 }, (_, i) => ({ t: i / FPS, pts: body({ x: 500, y: 600 + (i % 2) }) }))
    const result = findPhases(cleanTrack(frames)!, frames.map((f) => f.t))
    expect(result.ok).toBe(false)
  })
})

describe("analyzeFrames", () => {
  it("measures a 3 : 1 tempo and a still head as good", () => {
    const result = analyzeFrames(syntheticSwing(), "face-on", "right")
    expect(result.ok).toBe(true)
    if (!result.ok) return
    const tempo = result.analysis.metrics.find((m) => m.id === "tempo")!
    expect(tempo.status).toBe("good")
    expect(parseFloat(tempo.display)).toBeGreaterThan(2.5)
    expect(parseFloat(tempo.display)).toBeLessThan(3.6)
    expect(result.analysis.metrics.find((m) => m.id === "head-sway")!.status).toBe("good")
  })

  it("flags a head that slides toward the target", () => {
    const frames = syntheticSwing().map((f, i) => ({
      ...f,
      pts: body({ x: 500, y: f.pts![L.leftWrist].y }, { x: 500 + Math.max(0, i - 40) * 6, y: 200 }),
    }))
    const result = analyzeFrames(frames, "face-on", "right")
    expect(result.ok).toBe(true)
    if (!result.ok) return
    const head = result.analysis.metrics.find((m) => m.id === "head-sway")!
    expect(head.status).not.toBe("good")
    expect(head.fault).toBe("head-sway")
  })

  it("uses down-the-line checks for that view", () => {
    const result = analyzeFrames(syntheticSwing(), "down-the-line", "right")
    expect(result.ok).toBe(true)
    if (!result.ok) return
    const ids = result.analysis.metrics.map((m) => m.id)
    expect(ids).toContain("posture")
    expect(ids).not.toContain("hip-sway")
  })
})

describe("coach", () => {
  it("every fault the metrics can raise has a drill", () => {
    const faults = ["tempo-quick", "tempo-slow", "head-sway", "head-height", "hip-sway", "lead-arm-bent", "short-turn",
      "finish-balance", "posture-upright", "posture-hunched", "early-extension", "knee-flex"]
    for (const f of faults) expect(DRILLS.some((d) => d.faults.includes(f as never))).toBe(true)
  })

  it("puts setup problems before tempo", () => {
    const m = (id: string, fault: string, status: "ok" | "work") =>
      ({ id, label: id, display: "", target: "", status, why: "", fault, phase: "top" }) as never
    const plan = makePlan([m("tempo", "tempo-quick", "work"), m("posture", "posture-upright", "work")])
    expect(plan.first?.metric.id).toBe("posture")
    expect(plan.first?.drills.length).toBeGreaterThan(0)
  })
})

describe("the bundled sample swing (real MediaPipe output, face-on, slow motion)", () => {
  it("finds the phases in order with a believable tempo", async () => {
    const { readFileSync } = await import("node:fs")
    const frames = JSON.parse(readFileSync(new URL("./fixtures/sample-face-on.json", import.meta.url), "utf8")) as Frame[]
    const result = analyzeFrames(frames, "face-on", "right")
    expect(result.ok).toBe(true)
    if (!result.ok) return
    const { times, phases } = result.analysis
    // From the contact sheet: address before 19 s, top around 24–25 s, impact around 26.5 s
    expect(times[phases.address]).toBeLessThan(19.5)
    expect(times[phases.top]).toBeGreaterThan(23.5)
    expect(times[phases.top]).toBeLessThan(25.5)
    expect(times[phases.impact]).toBeGreaterThan(26)
    expect(times[phases.impact]).toBeLessThan(27.2)
    expect(times[phases.finish]).toBeGreaterThan(times[phases.impact])
    console.log(result.analysis.metrics.map((m) => `${m.id}: ${m.display} (${m.status})`).join("\n"))
  })
})

describe("the bundled sample swing, down the line", () => {
  it("finds the phases and uses the down-the-line checks", async () => {
    const { readFileSync } = await import("node:fs")
    const frames = JSON.parse(readFileSync(new URL("./fixtures/sample-down-the-line.json", import.meta.url), "utf8")) as Frame[]
    const result = analyzeFrames(frames, "down-the-line", "right")
    expect(result.ok).toBe(true)
    if (!result.ok) return
    const { times, phases, metrics } = result.analysis
    // Hands still until ~2.5 s, top around 6.5–7.5 s, lowest around 9.4 s
    expect(times[phases.address]).toBeLessThan(3.2)
    expect(times[phases.top]).toBeGreaterThan(6)
    expect(times[phases.top]).toBeLessThan(8)
    expect(times[phases.impact]).toBeGreaterThan(8.8)
    expect(times[phases.impact]).toBeLessThan(9.8)
    const ids = metrics.map((m) => m.id)
    expect(ids).toEqual(expect.arrayContaining(["tempo", "posture", "spine-kept", "hip-depth", "knee-flex"]))
    expect(ids).not.toContain("lead-arm")
    console.log(metrics.map((m) => `${m.id}: ${m.display} (${m.status})`).join("\n"))
  })
})
