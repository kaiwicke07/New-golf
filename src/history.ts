import type { Metric, Status } from "./analysis/metrics"
import type { Hand, View } from "./analysis/types"

/** A saved swing: its measurements and a small picture, kept only in this browser. */
export interface SavedSwing {
  id: string
  date: number
  name: string
  view: View
  hand: Hand
  thumb: string
  metrics: Pick<Metric, "id" | "label" | "display" | "status">[]
}

const KEY = "swingiq:history"
const LIMIT = 40

export function loadHistory(): SavedSwing[] {
  try {
    const raw = localStorage.getItem(KEY)
    return raw ? (JSON.parse(raw) as SavedSwing[]) : []
  } catch {
    return []
  }
}

export function saveSwing(swing: SavedSwing): SavedSwing[] {
  const list = [swing, ...loadHistory().filter((s) => s.id !== swing.id)].slice(0, LIMIT)
  try {
    localStorage.setItem(KEY, JSON.stringify(list))
  } catch {
    // Storage full or blocked (private mode): history just won't persist
  }
  return list
}

export function deleteSwing(id: string): SavedSwing[] {
  const list = loadHistory().filter((s) => s.id !== id)
  try {
    localStorage.setItem(KEY, JSON.stringify(list))
  } catch {
    // ignore
  }
  return list
}

export const countStatus = (metrics: { status: Status }[], status: Status) =>
  metrics.filter((m) => m.status === status).length
