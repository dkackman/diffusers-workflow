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
export function noteAnswered(note: LedgerNote, request: Standing): boolean {
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
  return !noteAnswered(note, request)
}

export type NoteLine = {
  text: string
  /** The agent has written since: the note may have done its work, so the
   * page offers to clear it */
  answered: boolean
}

/** What a waiting note, and the box a note is written in, say of when the
 * agent sees it: whether the agent is connected, which is whether its
 * notes channel is polling the workspace (listening.svelte.ts) */
export const LISTENING_LINE =
  'The agent is connected; it will see this within about 10 seconds.'
export const NOT_LISTENING_LINE =
  "The agent isn't connected. It reads this the next time it updates the plan."

/** A new note that has references but no words yet: the references say
 * what it is about, never what to change */
export const NEEDS_WORDS =
  'Add a few words: the items under About say what the note is about, not what to change.'

/** Where a note is on the request page, for a link to it */
export const noteAnchor = (id: string) => `request-note-${id}`

/** The line under a saved note: when it was saved (and edited), then
 * where it stands. `listening` is whether an agent session will hear a
 * waiting note now. */
export function noteLine(
  note: LedgerNote,
  request: StandingWithHistory,
  now: Date = new Date(),
  listening = false,
): NoteLine {
  const answered = noteAnswered(note, request)
  const edited =
    note.created !== note.at
      ? ` · edited ${formatMtime(seconds(note.at), now)}`
      : ''
  const saved = `Saved ${formatMtime(seconds(note.created), now)}${edited}.`
  if (request.state !== 'open')
    return {
      text: `${saved} This request is closed. The agent may not return to it.`,
      answered,
    }
  if (!answered)
    return {
      text: `${saved} ${listening ? LISTENING_LINE : NOT_LISTENING_LINE}`,
      answered,
    }
  const since = agentWritesSince(note, request.history)
  if (since.length === 0)
    return {
      text: `${saved} The agent has updated the plan since (${formatMtime(seconds(request.agent_wrote_at as string), now)}).`,
      answered,
    }
  // What it said is the header's and Plan changes' to show: here, only
  // that it has answered, and where to look
  const latest = since[since.length - 1]
  const times = since.length === 1 ? 'once' : `${since.length} times`
  return {
    text:
      `${saved} The agent has updated the plan ${times} since ` +
      `(latest r${latest.revision}, ${formatMtime(seconds(latest.at), now)}).`,
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
