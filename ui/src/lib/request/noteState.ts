// What the page may say about a saved note for the agent. The ledger holds
// the note's text and when it was written, and the agent's only voice is
// the `why` of its next write: so the page can say the agent has written
// since, never that it read the note.
import { formatMtime } from '../format'
import type { LedgerHistoryEntry, LedgerNote, LedgerRequest } from '../types'

const seconds = (at: string) => Date.parse(at) / 1000

/** The agent's writes after the note was saved, oldest first */
export function agentWritesSince(
  note: LedgerNote,
  history: LedgerHistoryEntry[],
): LedgerHistoryEntry[] {
  const saved = Date.parse(note.at)
  return history
    .filter((e) => e.by === 'agent' && Date.parse(e.at) > saved)
    .sort((a, b) => a.revision - b.revision)
}

type Standing = Pick<LedgerRequest, 'state' | 'agent_wrote_at'>
type StandingWithHistory = Standing & Pick<LedgerRequest, 'history'>

/** Whether the agent has written since the note was saved (a write in the
 * same second does not count: the ledger stamps whole seconds, so the note
 * still waits). An unparseable
 * time on either side leaves the note unanswered. */
function answeredBy(note: LedgerNote, request: Standing): boolean {
  if (!request.agent_wrote_at) return false
  const saved = Date.parse(note.at)
  const wrote = Date.parse(request.agent_wrote_at)
  if (Number.isNaN(saved) || Number.isNaN(wrote)) return false
  return saved < wrote
}

/** A note the agent has had no chance to read: on an open request, saved
 * after the agent's last write (`agent_wrote_at`). The same rule as
 * `note_waiting` in dw_mcp/ledger.py. */
export function noteWaiting(
  note: LedgerNote | null,
  request: Standing,
): boolean {
  if (!note || request.state !== 'open') return false
  return !answeredBy(note, request)
}

export type NoteLine = {
  text: string
  /** The agent has written since: the note may have done its work, so the
   * page offers to clear it */
  answered: boolean
}

/** The line under a saved note. */
export function noteLine(
  note: LedgerNote,
  request: StandingWithHistory,
  now: Date = new Date(),
): NoteLine {
  const answered = answeredBy(note, request)
  const saved = `Saved ${formatMtime(seconds(note.at), now)}.`
  if (request.state !== 'open')
    return {
      text: `${saved} This request is closed. The agent may not return to it.`,
      answered,
    }
  if (!answered)
    return {
      text: `${saved} Nothing is sent to the agent. It reads this the next time it updates the plan.`,
      answered,
    }
  const since = agentWritesSince(note, request.history)
  if (since.length === 0)
    return {
      text: `${saved} The agent has updated the plan since (${formatMtime(seconds(request.agent_wrote_at as string), now)}).`,
      answered,
    }
  const latest = since[since.length - 1]
  const times = since.length === 1 ? 'once' : `${since.length} times`
  return {
    text:
      `${saved} The agent has updated the plan ${times} since ` +
      `(latest r${latest.revision}, ${formatMtime(seconds(latest.at), now)}: “${latest.why}”).`,
    answered,
  }
}

/** How many notes wait on the agent: the request's own and each live item's */
export function notesWaiting(request: LedgerRequest): number {
  const notes = [
    ...request.notes,
    ...request.items.filter((item) => !item.dropped).flatMap((i) => i.notes),
  ]
  return notes.filter((note) => noteWaiting(note, request)).length
}
