// The draft of a note for the agent, on a request or one of its items:
// when leaving it loses something, and what a refused save says beside it.
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

/** The words beside a refused draft. A stale revision is said for the
 * person writing the note - the server's own sentence tells an agent to
 * rewrite its plan - and any other refusal is the server's message. */
export function refusalText(failure: {
  message: string
  reload: boolean
}): string {
  return failure.reload
    ? 'The plan changed while you were writing, so this is not saved yet. Reload the plan, then save again: your words stay here.'
    : failure.message
}

/** Where a note's unsaved draft is kept for the life of the tab, so a
 * reload does not lose it: by workspace, request and item (no item for the
 * request's own note). */
export const draftKey = (workspace: string, request: string, item?: string) =>
  `dw-request-note:${workspace}/${request}/${item ?? ''}`

export function readDraft(key: string | null): string | null {
  if (!key) return null
  try {
    return sessionStorage.getItem(key)
  } catch {
    return null
  }
}

/** Keep `text` under `key`; null forgets it. Storage that is missing or
 * full only means the draft does not outlive the page. */
export function keepDraft(key: string | null, text: string | null): void {
  if (!key) return
  try {
    if (text === null) sessionStorage.removeItem(key)
    else sessionStorage.setItem(key, text)
  } catch {
    /* session only */
  }
}
