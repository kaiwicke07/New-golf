// Copies MediaPipe's WebAssembly files next to the app, so pose tracking
// loads from this site and works offline, with no third-party CDN.
import { cpSync, mkdirSync } from "node:fs"

mkdirSync("public/mediapipe", { recursive: true })
cpSync("node_modules/@mediapipe/tasks-vision/wasm", "public/mediapipe/wasm", { recursive: true })
console.log("Copied MediaPipe wasm to public/mediapipe/wasm")
