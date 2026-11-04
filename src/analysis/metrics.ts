import { handsAt, torsoLength } from "./phases"
import { angleAt, dist, mid, tiltFromVertical } from "./series"
import { L, sides, type Hand, type Phases, type Point, type View } from "./types"

export type Status = "good" | "ok" | "work"

/** Things the coach knows how to fix; each maps to drills in coach/drills.ts. */
export type Fault =
  | "tempo-quick"
  | "tempo-slow"
  | "head-sway"
  | "head-height"
  | "hip-sway"
  | "lead-arm-bent"
  | "short-turn"
  | "finish-balance"
  | "posture-upright"
  | "posture-hunched"
  | "early-extension"
  | "knee-flex"

export interface Metric {
  id: string
  label: string
  /** The measured value, formatted in a sentence, e.g. "3.1 : 1" */
  display: string
  /** The value alone, short enough for a big readout, e.g. "20%" */
  value: string
  /** What the value is, e.g. "of shoulder width, toward the target" */
  unit: string
  /** The range we aim for, in words */
  target: string
  status: Status
  /** One plain sentence on what this measures and why it matters */
  why: string
  /** The fault to work on when status isn't good */
  fault: Fault | null
  /** The phase to show when this metric is selected */
  phase: keyof Phases
  /** Estimated from a 2D view; read as a guide */
  estimate?: boolean
}

interface Band {
  good: [number, number]
  ok: [number, number]
}
const within = (v: number, [lo, hi]: [number, number]) => v >= lo && v <= hi
const grade = (v: number, b: Band): Status => (within(v, b.good) ? "good" : within(v, b.ok) ? "ok" : "work")

/** Everything a metric needs: the cleaned track, the phases and the frame times. */
export interface SwingData {
  track: Point[][]
  times: number[]
  phases: Phases
  view: View
  hand: Hand
}

/**
 * Measures the swing. Each check uses only what the chosen camera view can
 * see: side-to-side movement from face-on, forward bend and hip depth from
 * down the line. Tempo and the lead arm work from either.
 */
export function measure({ track, times, phases, view, hand }: SwingData): Metric[] {
  const s = sides(hand)
  const at = (phase: keyof Phases) => track[phases[phase]]
  const torso = torsoLength(track) || 1
  const address = at("address")
  const top = at("top")
  const impact = at("impact")
  const finish = at("finish")
  const shoulderWidth = dist(address[L.leftShoulder], address[L.rightShoulder]) || torso * 0.6
  const metrics: Metric[] = []

  // Tempo: backswing time over downswing time. Tour players are close to 3 : 1.
  const back = times[phases.top] - times[phases.address]
  const down = times[phases.impact] - times[phases.top]
  const ratio = down > 0 ? back / down : 0
  const tempo = grade(ratio, { good: [2.5, 3.5], ok: [2, 4.2] })
  metrics.push({
    id: "tempo",
    value: ratio.toFixed(1) + " : 1",
    unit: "backswing to downswing",
    label: "Tempo",
    display: `${ratio.toFixed(1)} : 1`,
    target: "About 3 : 1 (backswing to downswing)",
    status: tempo,
    why: "Good players take about three times as long going back as coming down. A steady rhythm makes contact repeatable.",
    fault: tempo === "good" ? null : ratio < 2.5 ? "tempo-quick" : "tempo-slow",
    phase: "top",
  })

  // A slight dip is normal in good swings; rising is the bigger problem
  const headDrop = (impact[L.nose].y - address[L.nose].y) / torso
  const height = grade(headDrop, { good: [-0.05, 0.14], ok: [-0.1, 0.22] })
  metrics.push({
    id: "head-height",
    value: `${Math.round(Math.abs(headDrop) * 100)}%`,
    unit: `of torso length, ${headDrop > 0 ? "down" : "up"}`,
    label: "Head height",
    display: `${headDrop > 0 ? "Down" : "Up"} ${Math.round(Math.abs(headDrop) * 100)}% of torso`,
    target: "Steady: a small dip is fine, rising isn't",
    status: height,
    why: "If your head rises or dips between address and impact, the bottom of the swing moves too, which causes thin and fat shots.",
    fault: height === "good" ? null : "head-height",
    phase: "impact",
  })

  if (view === "face-on") {
    // Which way the target is on screen: the lead hip is on the target side at address
    const targetDir = Math.sign(address[s.leadHip].x - address[s.trailHip].x) || 1

    // Lead arm (face-on only: from down the line the arm points at the camera).
    // Measured when the hands first reach shoulder height going back.
    // At the very top the arm points partly at the camera and looks bent in 2D.
    let armFrame = phases.top
    for (let i = phases.address; i <= phases.top; i++) {
      if (handsAt(track[i]).y <= track[i][s.leadShoulder].y) {
        armFrame = i
        break
      }
    }
    const armPose = track[armFrame]
    const elbow = angleAt(armPose[s.leadShoulder], armPose[s.leadElbow], armPose[s.leadWrist])
    const arm = grade(elbow, { good: [150, 180], ok: [135, 180] })
    metrics.push({
      id: "lead-arm",
    value: `${Math.round(elbow)}°`,
    unit: "lead elbow, hands at shoulder height",
      label: "Lead arm in the backswing",
      display: `${Math.round(elbow)}°`,
      target: "150° or straighter as the hands pass your shoulder",
      status: arm,
      why: "A fairly straight lead arm in the backswing keeps the swing wide and the club on a consistent arc.",
      fault: arm === "good" ? null : "lead-arm-bent",
      phase: "top",
      estimate: true,
    })

    // Head height: bobbing up or down between address and impact changes the low point

    // Positive when the head moved toward the target by impact (the real fault);
    // staying a little behind the ball is normal
    const headShift = ((impact[L.nose].x - address[L.nose].x) * targetDir) / shoulderWidth
    const head = grade(headShift, { good: [-0.3, 0.12], ok: [-0.45, 0.25] })
    metrics.push({
      id: "head-sway",
    value: `${Math.round(Math.abs(headShift) * 100)}%`,
    unit: `of shoulder width, ${headShift > 0 ? "toward" : "away from"} the target`,
      label: "Head movement",
      display: `${Math.round(Math.abs(headShift) * 100)}% of shoulder width ${headShift > 0 ? "toward" : "away from"} the target`,
      target: "Head behind the ball: no more than 12% toward the target",
      status: head,
      why: "A steady head from address to impact keeps your swing centred, so the club comes back to the ball where it started.",
      fault: head === "good" ? null : "head-sway",
      phase: "impact",
    })

    const hipsAddress = mid(address[L.leftHip], address[L.rightHip])
    const hipsTop = mid(top[L.leftHip], top[L.rightHip])
    // Positive when the hips slid away from the target in the backswing
    const sway = ((hipsAddress.x - hipsTop.x) * targetDir) / shoulderWidth
    const hips = grade(sway, { good: [-1, 0.15], ok: [-1, 0.3] })
    metrics.push({
      id: "hip-sway",
    value: `${Math.round(Math.max(0, sway) * 100)}%`,
    unit: "of shoulder width, away from the target",
      label: "Hip sway in the backswing",
      display: `${Math.round(Math.max(0, sway) * 100)}% of shoulder width`,
      target: "Under 15% of your shoulder width",
      status: hips,
      why: "Your hips should turn in the backswing, not slide away from the target. Sliding makes it hard to get back to the ball.",
      fault: hips === "good" ? null : "hip-sway",
      phase: "top",
    })

    // Turned shoulders look narrower from the front; how much they narrow is a turn estimate
    const narrowing = dist(top[L.leftShoulder], top[L.rightShoulder]) / shoulderWidth
    const turn = grade(narrowing, { good: [0, 0.6], ok: [0, 0.78] })
    metrics.push({
      id: "turn",
    value: narrowing <= 0.6 ? "Full" : narrowing <= 0.78 ? "Partial" : "Short",
    unit: "estimated from the front",
      label: "Shoulder turn",
      display: narrowing <= 0.6 ? "Full" : narrowing <= 0.78 ? "Partial" : "Short",
      target: "A full turn: shoulders look much narrower at the top",
      status: turn,
      why: "A full shoulder turn stores power and gives your arms room to swing. From the front, turned shoulders look narrower on camera.",
      fault: turn === "good" ? null : "short-turn",
      phase: "top",
      estimate: true,
    })

    // Finish: the hips should end up over the lead foot
    const hipsFinish = mid(finish[L.leftHip], finish[L.rightHip])
    const overLead = Math.abs(hipsFinish.x - finish[s.leadAnkle].x) / shoulderWidth
    const balance = grade(overLead, { good: [0, 0.35], ok: [0, 0.6] })
    metrics.push({
      id: "finish",
    value: overLead <= 0.35 ? "Lead side" : overLead <= 0.6 ? "Forward" : "Back",
    unit: "where your hips finish",
      label: "Weight at the finish",
      display: overLead <= 0.35 ? "On your lead side" : overLead <= 0.6 ? "Mostly forward" : "Hanging back",
      target: "Hips stacked over your lead foot",
      status: balance,
      why: "Finishing with your weight on your lead foot shows you shifted through the ball instead of hanging back.",
      fault: balance === "good" ? null : "finish-balance",
      phase: "finish",
    })
  } else {
    // Down the line: forward bend, keeping it, knee flex, hips toward the ball
    const spine = (p: Point[]) =>
      tiltFromVertical(mid(p[L.leftHip], p[L.rightHip]), mid(p[L.leftShoulder], p[L.rightShoulder]))
    const setup = spine(address)
    const posture = grade(setup, { good: [28, 50], ok: [20, 58] })
    metrics.push({
      id: "posture",
    value: `${Math.round(setup)}°`,
    unit: "forward bend at address",
      label: "Posture at address",
      display: `${Math.round(setup)}° forward bend`,
      target: "Bent forward about 30–50° from the hips",
      status: posture,
      why: "Tilting forward from the hips at address gives your arms room to swing and sets the angle you'll swing on.",
      fault: posture === "good" ? null : setup < 28 ? "posture-upright" : "posture-hunched",
      phase: "address",
      estimate: true,
    })

    const lost = setup - spine(impact)
    const keep = grade(lost, { good: [-100, 6], ok: [-100, 11] })
    metrics.push({
      id: "spine-kept",
    value: `${Math.round(Math.max(0, lost))}°`,
    unit: "of forward bend lost by impact",
      label: "Posture kept through impact",
      display: lost > 0 ? `Stood up ${Math.round(lost)}°` : "Kept it",
      target: "Lose less than 6° of forward bend by impact",
      status: keep,
      why: "Standing up through the downswing (early extension) pushes the hips toward the ball and the club off its path.",
      fault: keep === "good" ? null : "early-extension",
      phase: "impact",
      estimate: true,
    })

    // Hips toward the ball: the ball is on the hands' side of the hips at address
    const ballDir = Math.sign(handsAt(address).x - mid(address[L.leftHip], address[L.rightHip]).x) || 1
    const hipForward =
      ((mid(impact[L.leftHip], impact[L.rightHip]).x - mid(address[L.leftHip], address[L.rightHip]).x) * ballDir) /
      torso
    const depth = grade(hipForward, { good: [-1, 0.08], ok: [-1, 0.16] })
    metrics.push({
      id: "hip-depth",
    value: `${Math.round(Math.max(0, hipForward) * 100)}%`,
    unit: "of torso length, toward the ball",
      label: "Hips toward the ball",
      display: `${Math.round(Math.max(0, hipForward) * 100)}% of torso`,
      target: "Under 8% of your torso length",
      status: depth,
      why: "Your hips should stay back as you turn through. Moving toward the ball crowds your arms and leads to blocks and shanks.",
      fault: depth === "good" ? null : "early-extension",
      phase: "impact",
    })

    const knee = angleAt(address[s.trailHip], address[s.trailKnee], address[s.trailAnkle])
    const flex = grade(knee, { good: [145, 170], ok: [135, 176] })
    metrics.push({
      id: "knee-flex",
    value: `${Math.round(180 - knee)}°`,
    unit: "of knee bend at address",
      label: "Knee flex at address",
      display: `${Math.round(180 - knee)}° of bend`,
      target: "A soft bend: about 10–35°",
      status: flex,
      why: "Slightly flexed knees let you turn and stay balanced. Locked or deeply bent knees make both harder.",
      fault: flex === "good" ? null : "knee-flex",
      phase: "address",
      estimate: true,
    })
  }

  return metrics
}
