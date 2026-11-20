// What the page may say about a saved note for the agent. The ledger holds
// the note's text and when it was written, and the agent's only voice is
// the `why` of its next write: so the page can say the agent has written
// since, never that it read the note.
import { formatMtime } from '../format'
import type { LedgerComment, LedgerHistoryEntry, LedgerRequest } from '../types'

const seconds = (at: string) => Date.parse(at) / 1000

/** The agent's writes after the note was saved, oldest first */
export function agentWritesSince(
  comment: LedgerComment,
  history: LedgerHistoryEntry[],
): LedgerHistoryEntry[] {
  const saved = Date.parse(comment.at)
  return history
    .filter((e) => e.by === 'agent' && Date.parse(e.at) > saved)
    .sort((a, b) => a.revision - b.revision)
}

type Standing = Pick<LedgerRequest, 'state' | 'history'>

/** A note the agent has had no chance to read: saved on an open request,
 * with no agent write after it. */
export function noteWaiting(
  comment: LedgerComment | null,
  request: Standing,
): boolean {
  if (!comment || request.state !== 'open') return false
  return agentWritesSince(comment, request.history).length === 0
}

export type NoteLine = {
  text: string
  /** The agent has written since: the note may have done its work, so the
   * page offers to clear it */
  answered: boolean
}

/** The line under a saved note. */
export function noteLine(
  comment: LedgerComment,
  request: Standing,
  now: Date = new Date(),
): NoteLine {
  const since = agentWritesSince(comment, request.history)
  const answered = since.length > 0
  const saved = `Saved ${formatMtime(seconds(comment.at), now)}.`
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
    request.comment,
    ...request.items.filter((item) => !item.dropped).map((i) => i.comment),
  ]
  return notes.filter((comment) => noteWaiting(comment, request)).length
}
