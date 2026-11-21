// The request's history as the rail lists it: newest first, with a run of
// the user's own plain writes folded into one line, so the agent's last
// sentence is not buried under "comment on a", "comment on b".
import type { LedgerHistoryEntry } from '../types'

export type Change =
  | { kind: 'entry'; entry: LedgerHistoryEntry }
  | { kind: 'user-run'; entries: LedgerHistoryEntry[] }

/** A user write that changed neither the goal nor the state */
const plainUser = (entry: LedgerHistoryEntry) =>
  entry.by === 'user' && !entry.goal && !entry.state

/** Newest first; two or more consecutive plain user writes become one
 * `user-run`, its entries newest first too. */
export function foldChanges(history: LedgerHistoryEntry[]): Change[] {
  const newest = [...history].sort((a, b) => b.revision - a.revision)
  const changes: Change[] = []
  let run: LedgerHistoryEntry[] = []
  const flush = () => {
    if (run.length === 1) changes.push({ kind: 'entry', entry: run[0] })
    else if (run.length) changes.push({ kind: 'user-run', entries: run })
    run = []
  }
  for (const entry of newest) {
    if (plainUser(entry)) run.push(entry)
    else {
      flush()
      changes.push({ kind: 'entry', entry })
    }
  }
  flush()
  return changes
}

/** What a history entry's `state` says happened, or null when it carries
 * none or it is the request being opened in the first place (`first`). */
export function stateChange(
  entry: LedgerHistoryEntry,
  first: boolean,
): string | null {
  if (!entry.state) return null
  if (entry.state === 'open') return first ? null : 'reopened'
  return entry.state === 'done' ? 'marked done' : 'abandoned'
}

export const who = (entry: LedgerHistoryEntry) =>
  entry.by === 'user' ? 'you' : 'agent'
