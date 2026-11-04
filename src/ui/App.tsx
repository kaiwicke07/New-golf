import { useEffect, useState } from "react"

import { DrillsPage } from "./DrillsPage"
import { HistoryPage } from "./HistoryPage"
import { StartScreen } from "./StartScreen"
import { SAMPLE_PARTS } from "./samples"
import { Workspace, type Source } from "./Workspace"

type Tab = "analyze" | "drills" | "history"

const SAMPLE: Source = {
  url: `${import.meta.env.BASE_URL}sample/sample-swing.webm`,
  name: "Sample swing",
  sample: SAMPLE_PARTS[0],
}

/** The page is chosen by the URL hash (#drills, #history) so the back button works. */
function tabFromHash(): Tab {
  const h = window.location.hash.slice(1)
  return h === "drills" || h === "history" ? h : "analyze"
}

export function App() {
  const [tab, setTab] = useState<Tab>(tabFromHash)
  const [source, setSource] = useState<Source | null>(null)
  const [drillFilter, setDrillFilter] = useState<string | null>(null)

  useEffect(() => {
    const onHash = () => setTab(tabFromHash())
    window.addEventListener("hashchange", onHash)
    return () => window.removeEventListener("hashchange", onHash)
  }, [])

  const go = (next: Tab) => {
    window.location.hash = next === "analyze" ? "" : next
    setTab(next)
    window.scrollTo({ top: 0 })
  }

  // Object URLs hold the whole video in memory; release them when a new one replaces it
  useEffect(() => {
    return () => {
      if (source && !source.sample) URL.revokeObjectURL(source.url)
    }
  }, [source])

  return (
    <>
      <header className="header">
        <div className="header-inner">
          <a
            className="brand"
            href="#"
            onClick={(e) => {
              e.preventDefault()
              setSource(null)
              go("analyze")
            }}
          >
            <span className="brand-mark" aria-hidden="true" />
            SWINGIQ
          </a>
          <nav className="nav" aria-label="Main">
            <button type="button" aria-current={tab === "analyze" ? "page" : undefined} onClick={() => go("analyze")}>
              Analyze
            </button>
            <button type="button" aria-current={tab === "drills" ? "page" : undefined} onClick={() => go("drills")}>
              Drills
            </button>
            <button type="button" aria-current={tab === "history" ? "page" : undefined} onClick={() => go("history")}>
              My swings
            </button>
          </nav>
          <span className="status">On-device · free</span>
        </div>
      </header>

      <main className="page">
        {tab === "analyze" &&
          (source ? (
            <Workspace
              key={source.url}
              source={source}
              onReset={() => setSource(null)}
              onShowDrills={(fault) => {
                setDrillFilter(fault)
                go("drills")
              }}
            />
          ) : (
            <StartScreen onPick={setSource} onSample={() => setSource(SAMPLE)} />
          ))}
        {tab === "drills" && <DrillsPage initialFault={drillFilter} />}
        {tab === "history" && <HistoryPage onAnalyze={() => go("analyze")} />}
      </main>

      <footer className="footer">
        <span>SwingIQ runs MediaPipe pose tracking in your browser. Measurements from one camera are estimates, not a lesson.</span>
        <span>
          Sample clip: &ldquo;Suvichaya Vinijchaitham Golf Swing Slow Mo 2026&rdquo; by Hut2026, CC0, via{" "}
          <a href="https://commons.wikimedia.org/wiki/File:Suvichaya_Vinijchaitham_Golf_Swing_Slow_Mo_2026.webm" target="_blank" rel="noopener">
            Wikimedia Commons
          </a>
        </span>
      </footer>
    </>
  )
}
