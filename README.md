# SwingIQ

Free golf swing analysis that runs entirely in your browser. Upload or record a video of your swing; SwingIQ tracks your body frame by frame, finds the four key moments (address, top, impact, finish), measures your swing, tells you the one thing to fix first, and gives you drills to fix it.

**Private by design:** the video is analysed on your device with MediaPipe pose tracking. Nothing is uploaded, there's no account, and no AI service is involved.

## What it measures

| Check | Camera view | What it means |
|---|---|---|
| Tempo | both | Backswing time over downswing time; good players are close to 3 : 1 |
| Head height | both | Rising or dipping between address and impact |
| Lead arm | face-on | Elbow angle as the hands pass shoulder height |
| Head movement | face-on | Head sliding toward the target by impact |
| Hip sway | face-on | Hips sliding away from the target in the backswing |
| Shoulder turn | face-on | Estimated from how much the shoulders narrow on camera at the top |
| Weight at the finish | face-on | Hips stacked over the lead foot |
| Posture at address | down the line | Forward bend from the hips |
| Posture kept | down the line | Standing up through impact (early extension) |
| Hips toward the ball | down the line | Hips moving toward the ball by impact |
| Knee flex | down the line | Knee bend at address |

Each check has a target range; anything outside it maps to drills in a 14-drill library, ordered so setup problems come before movement, rhythm and finish. Measurements from one 2D camera are estimates and are labelled as such.

## How it works

1. **Pose tracking** (`src/pose/detect.ts`): MediaPipe's pose landmarker (full model, served from this site) runs on each sampled frame, up to 240 frames per clip, on the GPU when available.
2. **Cleaning** (`src/analysis/series.ts`): low-confidence landmarks are interpolated, then smoothed with a short centred moving average.
3. **Phases** (`src/analysis/phases.ts`): impact is near the fastest hand speed; address is the last still moment with the hands low before that; the top is the highest the hands go in between; the finish is the highest after impact. Stillness is relative to the clip's peak speed, so slow-motion video works.
4. **Measurements** (`src/analysis/metrics.ts`): angles and distances normalised by the golfer's own torso length or shoulder width, so camera distance doesn't matter.
5. **Coaching** (`src/coach/`): a priority order picks the first fault to fix; drills are written out step by step.

History is kept in the browser's local storage.

## Development

```bash
npm install      # also copies MediaPipe's wasm into public/mediapipe
npm run dev
npm test         # includes two real swings from the sample clip as fixtures
npm run build
```

## Credits

- Pose tracking: [MediaPipe](https://developers.google.com/mediapipe) (Apache 2.0)
- Sample clip: "Suvichaya Vinijchaitham Golf Swing Slow Mo 2026" by Hut2026, CC0, via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Suvichaya_Vinijchaitham_Golf_Swing_Slow_Mo_2026.webm)
