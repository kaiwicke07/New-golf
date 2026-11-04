import { useEffect, useRef, useState } from "react"

import { warmUp } from "../pose/detect"
import { Camera, PlayIcon, Upload, ViewDiagram } from "./icons"
import type { Source } from "./Workspace"

/** The first screen: what this is, what it measures, how to film, and a way in. */
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
    onPick({ url: URL.createObjectURL(file), name: file.name.replace(/\.[^.]+$/, "") })
  }

  return (
    <div className="start">
      <div className="start-hero">
        <div>
          <h1>
            Read your swing <em>like a pro.</em>
          </h1>
        </div>
        <div>
          <p className="start-lead">
            Film a swing on your phone. SwingIQ tracks your body frame by frame, measures tempo, posture, sway and
            balance, and tells you the one thing to fix first, with drills to fix it.
          </p>
          <div className="start-actions">
            <button type="button" className="btn btn-primary" onClick={() => fileInput.current?.click()}>
              <Upload /> Choose a video
            </button>
            <button type="button" className="btn" onClick={() => cameraInput.current?.click()}>
              <Camera /> Record
            </button>
            <button type="button" className="btn btn-ghost" onClick={onSample}>
              <PlayIcon /> Try a sample swing
            </button>
          </div>
          <input ref={fileInput} type="file" accept="video/*" hidden onChange={(e) => pick(e.target.files?.[0])} />
          <input ref={cameraInput} type="file" accept="video/*" capture="environment" hidden onChange={(e) => pick(e.target.files?.[0])} />
          {error && (
            <p className="error" role="alert" style={{ marginTop: 16 }}>
              {error}
            </p>
          )}
        </div>
      </div>

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
        Drop a swing video here
      </div>

      <div className="specs">
        <div className="spec">
          <div className="spec-value">$0</div>
          <p>Free, with no account and no limits</p>
        </div>
        <div className="spec">
          <div className="spec-value">0 MB</div>
          <p>Uploaded. Your video is analysed on this device</p>
        </div>
        <div className="spec">
          <div className="spec-value">11</div>
          <p>Checks, from tempo and head movement to posture and balance</p>
        </div>
        <div className="spec">
          <div className="spec-value">14</div>
          <p>Drills, written out step by step</p>
        </div>
      </div>

      <section className="film" aria-labelledby="film-title">
        <div className="film-view">
          <ViewDiagram view="face-on" />
          <h3>Face-on</h3>
          <p>In front of you, facing your chest. Best for sway, head movement, turn and balance.</p>
        </div>
        <div className="film-view">
          <ViewDiagram view="down-the-line" />
          <h3>Down the line</h3>
          <p>Behind your hands, looking at the target. Best for posture and early extension.</p>
        </div>
        <div>
          <h2 id="film-title">How to film</h2>
          <ol>
            <li>Prop the phone at hip height, about 3 metres away. Don&rsquo;t hold it.</li>
            <li>Get your whole body in frame, head to feet, with room above for the club.</li>
            <li>Start a second before you swing; stop after you hold the finish.</li>
            <li>Good light helps. Slow motion is fine.</li>
          </ol>
        </div>
      </section>
    </div>
  )
}
