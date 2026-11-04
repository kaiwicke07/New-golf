import { useEffect, useRef, useState } from "react"

import { warmUp } from "../pose/detect"
import { Camera, Check, PlayIcon, Upload, ViewDiagram } from "./icons"
import type { Source } from "./Workspace"

/** The first screen: what this is, how to film, and a way in. */
export function StartScreen({ onPick, onSample }: { onPick: (s: Source) => void; onSample: () => void }) {
  const fileInput = useRef<HTMLInputElement>(null)
  const cameraInput = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Start fetching the pose model while the visitor reads, so analysis starts sooner
  useEffect(() => warmUp(), [])

  const pick = (file: File | undefined) => {
    if (!file) return
    if (!file.type.startsWith("video/")) {
      setError("That file isn't a video. Choose an MP4, MOV or WebM file.")
      return
    }
    onPick({ url: URL.createObjectURL(file), name: file.name })
  }

  return (
    <div className="start">
      <div>
        <h1>See what your swing is really doing.</h1>
        <p className="start-lead">
          Upload a video of your golf swing. SwingIQ tracks your body frame by frame, measures your tempo, posture and
          sway, and tells you the one thing to fix first, with drills to fix it.
        </p>

        <div className="start-actions">
          <button type="button" className="btn btn-primary" onClick={() => fileInput.current?.click()}>
            <Upload /> Choose a video
          </button>
          <button type="button" className="btn btn-secondary" onClick={() => cameraInput.current?.click()}>
            <Camera /> Record a swing
          </button>
          <button type="button" className="btn btn-ghost" onClick={onSample}>
            <PlayIcon /> Try a sample swing
          </button>
        </div>
        <input ref={fileInput} type="file" accept="video/*" hidden onChange={(e) => pick(e.target.files?.[0])} />
        <input ref={cameraInput} type="file" accept="video/*" capture="environment" hidden onChange={(e) => pick(e.target.files?.[0])} />

        <div
          className={`drop ${over ? "is-over" : ""}`}
          onDragOver={(e) => {
            e.preventDefault()
            setOver(true)
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setOver(false)
            pick(e.dataTransfer.files[0])
          }}
        >
          Or drop a video file here
        </div>
        {error && (
          <p className="error" role="alert" style={{ marginTop: 16 }}>
            {error}
          </p>
        )}

        <ul className="checks-list">
          <li>
            <Check /> Free, with no account and no limits
          </li>
          <li>
            <Check /> Private: the video is analysed on your device and never uploaded
          </li>
          <li>
            <Check /> Measures tempo, head movement, hip sway, posture, lead arm, turn and balance
          </li>
          <li>
            <Check /> A drill library with step-by-step instructions
          </li>
        </ul>
      </div>

      <section className="film" aria-labelledby="film-title">
        <h2 id="film-title">How to film your swing</h2>
        <div className="film-views">
          <div className="film-view">
            <ViewDiagram view="face-on" />
            <h3>Face-on</h3>
            <p>In front of you, facing your chest. Best for sway, head movement and balance.</p>
          </div>
          <div className="film-view">
            <ViewDiagram view="down-the-line" />
            <h3>Down the line</h3>
            <p>Behind your hands, looking at the target. Best for posture and early extension.</p>
          </div>
        </div>
        <ol>
          <li>Prop the phone at hip height, about 3 metres away, and don&rsquo;t hold it.</li>
          <li>Get your whole body in the frame, head to feet, with a little room above the club.</li>
          <li>Start recording a second before you swing and stop after you hold your finish.</li>
          <li>Good light helps, and slow motion is fine.</li>
        </ol>
      </section>
    </div>
  )
}
