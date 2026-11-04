import { useState } from "react"

import type { Fault } from "../analysis/metrics"
import { DRILLS, type Drill } from "../coach/drills"

const GROUPS: { id: string; label: string; faults: Fault[] }[] = [
  { id: "all", label: "All drills", faults: [] },
  { id: "tempo", label: "Tempo", faults: ["tempo-quick", "tempo-slow"] },
  { id: "head", label: "Head", faults: ["head-sway", "head-height"] },
  { id: "hips", label: "Hips and sway", faults: ["hip-sway", "early-extension"] },
  { id: "arms", label: "Arms and turn", faults: ["lead-arm-bent", "short-turn"] },
  { id: "setup", label: "Setup", faults: ["posture-upright", "posture-hunched", "knee-flex"] },
  { id: "finish", label: "Balance and finish", faults: ["finish-balance"] },
]

export function DrillCard({ drill }: { drill: Drill }) {
  return (
    <article className="card drill">
      <div className="drill-head">
        <h3>{drill.name}</h3>
      </div>
      <p className="muted">Fixes: {drill.fixes.toLowerCase()}</p>
      <ol>
        {drill.steps.map((step) => (
          <li key={step}>{step}</li>
        ))}
      </ol>
      <div className="drill-meta">
        <span>
          <b>Reps:</b> {drill.reps}
        </span>
        <span>
          <b>You need:</b> {drill.kit}
        </span>
      </div>
      <p className="feel">Feels right when: {drill.feel}</p>
    </article>
  )
}

/** The free drill library, filterable by what you're working on. */
export function DrillsPage({ initialFault }: { initialFault: string | null }) {
  const [group, setGroup] = useState(
    () => GROUPS.find((g) => g.faults.includes(initialFault as Fault))?.id ?? "all",
  )
  const active = GROUPS.find((g) => g.id === group)!
  const drills = active.faults.length ? DRILLS.filter((d) => d.faults.some((f) => active.faults.includes(f))) : DRILLS

  return (
    <>
      <div className="page-head">
        <h1>Drills</h1>
        <p>
          {DRILLS.length} practice drills, each written out step by step. After you analyse a swing, SwingIQ points
          you to the ones for your biggest fault.
        </p>
      </div>
      <div className="chips" role="group" aria-label="Filter drills">
        {GROUPS.map((g) => (
          <button key={g.id} type="button" aria-pressed={g.id === group} onClick={() => setGroup(g.id)}>
            {g.label}
          </button>
        ))}
      </div>
      <div className="drill-grid">
        {drills.map((d) => (
          <DrillCard key={d.id} drill={d} />
        ))}
      </div>
    </>
  )
}
