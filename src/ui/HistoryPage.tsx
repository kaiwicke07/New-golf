import { useState } from "react"

import { countStatus, deleteSwing, loadHistory } from "../history"
import { Trash } from "./icons"

/** Past analyses, kept in this browser only. */
export function HistoryPage({ onAnalyze }: { onAnalyze: () => void }) {
  const [list, setList] = useState(loadHistory)

  return (
    <>
      <div className="page-head">
        <h1>My swings</h1>
        <p>Every swing you analyse is saved here, on this device only, so you can see whether a fix is sticking.</p>
      </div>

      {list.length === 0 ? (
        <div className="empty">
          <h2>No swings yet</h2>
          <p>Analyse a swing and its measurements will show up here.</p>
          <button type="button" className="btn btn-primary" onClick={onAnalyze}>
            Analyse a swing
          </button>
        </div>
      ) : (
        <ul className="history">
          {list.map((s) => (
            <li key={s.id} className="history-item">
              {s.thumb ? <img src={s.thumb} alt="" /> : <span />}
              <div>
                <h3>{s.name}</h3>
                <p className="muted">
                  {new Date(s.date).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })} ·{" "}
                  {s.view === "face-on" ? "Face-on" : "Down the line"} · {countStatus(s.metrics, "good")} good,{" "}
                  {countStatus(s.metrics, "work") + countStatus(s.metrics, "ok")} to work on
                </p>
                <div className="history-metrics">
                  {s.metrics.map((m) => (
                    <span key={m.id} className={`is-${m.status}`}>
                      {m.label}: {m.display}
                    </span>
                  ))}
                </div>
              </div>
              <button
                type="button"
                className="btn btn-small"
                onClick={() => setList(deleteSwing(s.id))}
                aria-label={`Delete ${s.name}`}
              >
                <Trash /> Delete
              </button>
            </li>
          ))}
        </ul>
      )}
    </>
  )
}
