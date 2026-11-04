/** Small inline icons (24px grid, stroke follows the text colour). */
const base = { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": true } as const

export const Upload = () => (
  <svg {...base}><path d="M12 16V4M6 10l6-6 6 6M4 20h16" /></svg>
)
export const Camera = () => (
  <svg {...base}><path d="M3 8h3l2-3h8l2 3h3v11H3z" /><circle cx="12" cy="13" r="4" /></svg>
)
export const PlayIcon = () => (
  <svg {...base} fill="currentColor" stroke="none"><path d="M7 4.5v15l12-7.5z" /></svg>
)
export const PauseIcon = () => (
  <svg {...base} fill="currentColor" stroke="none"><rect x="6" y="4.5" width="4" height="15" rx="1" /><rect x="14" y="4.5" width="4" height="15" rx="1" /></svg>
)
export const StepBack = () => (
  <svg {...base}><path d="M15 6l-6 6 6 6" /></svg>
)
export const StepForward = () => (
  <svg {...base}><path d="M9 6l6 6-6 6" /></svg>
)
export const Check = () => (
  <svg {...base}><path d="M5 12.5l4.5 4.5L19 7.5" /></svg>
)
export const Lock = () => (
  <svg {...base}><rect x="5" y="11" width="14" height="9" rx="2" /><path d="M8 11V8a4 4 0 0 1 8 0v3" /></svg>
)
export const Trash = () => (
  <svg {...base}><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13" /></svg>
)

/** The brand mark: a flag in a hole. */
export const Flag = () => (
  <svg viewBox="0 0 32 32" aria-hidden="true">
    <rect width="32" height="32" rx="9" fill="#17663f" />
    <path d="M12 25V7l10 4-10 4" fill="#f2b705" stroke="#f2b705" strokeWidth="1.5" strokeLinejoin="round" />
    <path d="M12 25V7" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" />
    <ellipse cx="12" cy="25.5" rx="5" ry="1.4" fill="#fff" opacity=".55" />
  </svg>
)

/** Where to put the camera, seen from above: golfer, ball, target line and phone. */
export function ViewDiagram({ view }: { view: "face-on" | "down-the-line" }) {
  const faceOn = view === "face-on"
  return (
    <svg viewBox="0 0 160 100" aria-hidden="true">
      <line x1="10" y1="40" x2="150" y2="40" stroke="#17663f" strokeWidth="2" strokeDasharray="5 5" />
      <text x="150" y="33" textAnchor="end" fontSize="10" fill="#5b6862" fontFamily="inherit">Target</text>
      <polygon points="152,40 144,36 144,44" fill="#17663f" />
      <circle cx="80" cy="40" r="4" fill="#fff" stroke="#12201a" strokeWidth="1.5" />
      <ellipse cx="80" cy="62" rx="16" ry="9" fill="#12201a" />
      <circle cx="80" cy="62" r="6" fill="#5b6862" />
      {faceOn ? (
        <g>
          <rect x="72" y="4" width="16" height="10" rx="2" fill="#f2b705" stroke="#3a2c00" strokeWidth="1.2" />
          <path d="M80 16 L80 30" stroke="#3a2c00" strokeWidth="1.5" />
        </g>
      ) : (
        <g>
          <rect x="14" y="56" width="10" height="16" rx="2" fill="#f2b705" stroke="#3a2c00" strokeWidth="1.2" />
          <path d="M26 64 L60 62" stroke="#3a2c00" strokeWidth="1.5" />
        </g>
      )}
    </svg>
  )
}
