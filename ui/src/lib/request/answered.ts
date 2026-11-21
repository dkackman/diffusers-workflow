// The agent answered: a note that was waiting when the page last read the
// plan and that the agent has written since. The page reads it off two
// reads of the plan; nothing is sent to it.
import type { LedgerNote, LedgerRequest } from '../types'
import { noteAnswered, noteWaiting } from './noteState'

const notesOf = (request: LedgerRequest): LedgerNote[] => [
  ...request.notes,
  ...request.items.flatMap((item) => item.notes),
]

/** The notes waiting in `before` that `after` shows answered: the same
 * note, unedited, with an agent write later than it. A note edited or
 * deleted between the two is not one the agent answered. */
export function answeredNow(
  before: LedgerRequest,
  after: LedgerRequest,
): LedgerNote[] {
  const waiting = new Map(
    notesOf(before)
      .filter((note) => noteWaiting(note, before))
      .map((note) => [note.id, note.at]),
  )
  return notesOf(after).filter(
    (note) => waiting.get(note.id) === note.at && noteAnswered(note, after),
  )
}

/** The item a note is on; null for the request's own */
export function noteItem(request: LedgerRequest, id: string): string | null {
  const home = request.items.find((item) => item.notes.some((n) => n.id === id))
  return home?.key ?? null
}

/** The toast: what the agent last wrote, in its own words */
export function answeredText(request: LedgerRequest): string {
  const latest = request.history
    .filter((entry) => entry.by === 'agent')
    .sort((a, b) => a.revision - b.revision)
    .at(-1)
  const head = `The agent updated ${request.title} after your note`
  return latest ? `${head}: “${latest.why}”` : `${head}.`
}
