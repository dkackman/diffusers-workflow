// The comment draft on a request item: when leaving it loses something,
// and what a refused save says beside it.
import { ApiError } from '../apiErrors'

/** Whether closing the editor would lose words: the draft has text, and it
 * is not the comment already saved. */
export function draftAtRisk(draft: string, saved: string | null): boolean {
  return draft.trim() !== '' && draft !== (saved ?? '')
}

/** What the item shows under the draft when a save is refused: the
 * server's own message, and `reload` when reading the request again is the
 * answer - a `stale_revision` only; a workspace with no root folder
 * (`no_workspace_root`) or an unreadable ledger is not cured by a reload.
 * Null for a failure that is not the server refusing the write (the
 * network, a 5xx), which is not about this draft. */
export function saveFailure(
  error: unknown,
): { message: string; reload: boolean } | null {
  if (!(error instanceof ApiError) || error.status >= 500) return null
  return { message: error.message, reload: error.code === 'stale_revision' }
}
