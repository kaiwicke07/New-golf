import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision"

import type { Frame } from "../analysis/types"

/** Served from this site (copied by scripts/copy-mediapipe.mjs), so nothing loads from a third party. */
const WASM_PATH = `${import.meta.env.BASE_URL}mediapipe/wasm`
const MODEL_PATH = `${import.meta.env.BASE_URL}models/pose_landmarker_full.task`

/** At most this many frames are analysed per clip, so long or slow-motion clips stay quick. */
const MAX_FRAMES = 240
const MAX_FPS = 30

let landmarker: Promise<PoseLandmarker> | null = null

/** Loads the pose model once, on the GPU when the browser allows it. */
function getLandmarker(): Promise<PoseLandmarker> {
  landmarker ??= (async () => {
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH)
    const options = (delegate: "GPU" | "CPU") => ({
      baseOptions: { modelAssetPath: MODEL_PATH, delegate },
      runningMode: "VIDEO" as const,
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    })
    try {
      return await PoseLandmarker.createFromOptions(vision, options("GPU"))
    } catch {
      return await PoseLandmarker.createFromOptions(vision, options("CPU"))
    }
  })()
  landmarker.catch(() => (landmarker = null))
  return landmarker
}

/** Starts downloading the model before it's needed (e.g. when the upload screen opens). */
export const warmUp = () => void getLandmarker().catch(() => {})

function seek(video: HTMLVideoElement, t: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const done = () => {
      video.removeEventListener("seeked", done)
      video.removeEventListener("error", fail)
      resolve()
    }
    const fail = () => reject(new Error("The video couldn't be read."))
    video.addEventListener("seeked", done)
    video.addEventListener("error", fail)
    video.currentTime = t
  })
}

/**
 * Steps through [start, end] of the video and finds the golfer's pose in each
 * frame. Frame positions are in video pixels. Runs entirely in the browser.
 */
export async function trackPose(
  video: HTMLVideoElement,
  start: number,
  end: number,
  onProgress: (done: number, total: number, frame: Frame) => void,
  signal: AbortSignal,
): Promise<{ frames: Frame[]; fps: number }> {
  const model = await getLandmarker()
  const duration = Math.max(0.1, end - start)
  const fps = Math.min(MAX_FPS, MAX_FRAMES / duration)
  const total = Math.max(2, Math.floor(duration * fps))
  const w = video.videoWidth
  const h = video.videoHeight
  const frames: Frame[] = []

  for (let i = 0; i < total; i++) {
    if (signal.aborted) throw new DOMException("Cancelled", "AbortError")
    const t = start + i / fps
    await seek(video, t)
    // MediaPipe needs increasing timestamps; wall-clock time always increases, even across clips
    const result = model.detectForVideo(video, performance.now())
    const pose = result.landmarks[0]
    const frame: Frame = {
      t,
      pts: pose ? pose.map((p) => ({ x: p.x * w, y: p.y * h, v: p.visibility ?? 1 })) : null,
    }
    frames.push(frame)
    onProgress(i + 1, total, frame)
  }
  return { frames, fps }
}
