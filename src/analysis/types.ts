/** A body point in video pixels; `v` is MediaPipe's visibility (0–1). */
export interface Point {
  x: number
  y: number
  v: number
}

/** One analysed video frame: its time and the 33 MediaPipe pose landmarks, or null if no person was found. */
export interface Frame {
  t: number
  pts: Point[] | null
}

/** Where the camera stood. Face-on films the chest; down the line films from behind the hands, along the target line. */
export type View = "face-on" | "down-the-line"
export type Hand = "right" | "left"

/** MediaPipe pose landmark indices (the subject's own left and right). */
export const L = {
  nose: 0,
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
  leftKnee: 25,
  rightKnee: 26,
  leftAnkle: 27,
  rightAnkle: 28,
} as const

/** Bones drawn on the overlay. */
export const BONES: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28],
  [27, 31], [28, 32], [27, 29], [28, 30],
]

/** The golfer's lead side is the one nearer the target: the left for a right-handed golfer. */
export function sides(hand: Hand) {
  const lead = hand === "right" ? "left" : "right"
  const trail = hand === "right" ? "right" : "left"
  const pick = (side: "left" | "right", part: "Shoulder" | "Elbow" | "Wrist" | "Hip" | "Knee" | "Ankle") =>
    L[`${side}${part}` as keyof typeof L]
  return {
    leadShoulder: pick(lead, "Shoulder"),
    leadElbow: pick(lead, "Elbow"),
    leadWrist: pick(lead, "Wrist"),
    leadHip: pick(lead, "Hip"),
    leadKnee: pick(lead, "Knee"),
    leadAnkle: pick(lead, "Ankle"),
    trailShoulder: pick(trail, "Shoulder"),
    trailHip: pick(trail, "Hip"),
    trailKnee: pick(trail, "Knee"),
    trailAnkle: pick(trail, "Ankle"),
  }
}

export interface Phases {
  address: number
  top: number
  impact: number
  finish: number
}

export const PHASE_LABELS: Record<keyof Phases, string> = {
  address: "Address",
  top: "Top",
  impact: "Impact",
  finish: "Finish",
}
