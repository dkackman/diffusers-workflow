// The draft of a note for the agent, on a request or one of its items:
// when leaving it loses something, and what a refused save says beside it.
import { ApiError } from '../apiErrors'
import type { LedgerNoteRef } from '../types'

/** Whether closing the editor would lose words: the draft has text, and it
 * is not the note already saved. */
export function draftAtRisk(draft: string, saved: string | null): boolean {
  return draft.trim() !== '' && draft !== (saved ?? '')
}

export type SaveFailure = {
  message: string
  reload: boolean
  /** The server's code, null when it sent none */
  code: string | null
}

/** Refusals that reading the request again answers: the plan moved
 * (`stale_revision`), the note was deleted elsewhere (`unknown_note`), or
 * the item it is on was dropped or removed (`item_dropped`,
 * `unknown_item`). */
const RELOADS = new Set([
  'stale_revision',
  'unknown_note',
  'item_dropped',
  'unknown_item',
])

/** What the item shows under the draft when a save is refused: the
 * server's own message and code, and `reload` when reading the request
 * again is the answer; a workspace with no root folder
 * (`no_workspace_root`), a bad reference or a full list is not cured by a
 * reload. Null for a failure that is not the server refusing the write
 * (the network, a 5xx), which is not about this draft. */
export function saveFailure(error: unknown): SaveFailure | null {
  if (!(error instanceof ApiError) || error.status >= 500) return null
  const code = error.code ?? null
  return { message: error.message, reload: RELOADS.has(code ?? ''), code }
}

// The box goes with the item when the plan is read again: the words are
// the user's to copy first
const GONE_ITEM =
  'This item was dropped or removed from the plan, so the note is not saved. Copy your words first: reloading the plan closes this box.'

/** The words beside a refused draft, said for the person writing the note
 * where the server's sentence is for an agent (a stale revision tells it
 * to rewrite its plan) or names no way on; any other refusal is the
 * server's message. */
export function refusalText(failure: SaveFailure): string {
  switch (failure.code) {
    case 'stale_revision':
      return 'The plan changed while you were writing, so this is not saved yet. Reload the plan, then save again: your words stay here.'
    case 'unknown_note':
      return 'This note is no longer there: it was deleted somewhere else. Reload the plan; your words stay here.'
    case 'item_dropped':
    case 'unknown_item':
      return GONE_ITEM
    case 'invalid_note_ref':
      return `A take or file this note refers to does not hold any more (${failure.message}). Remove that reference and save again.`
    default:
      return failure.message
  }
}

/** A note's references as one string for its draft's key: each as
 * `item/job/output`, joined with `,`; "" for none. */
export const refKey = (refs: LedgerNoteRef[]) =>
  refs.map((r) => `${r.item}/${r.job ?? ''}/${r.output ?? ''}`).join(',')

/** Where a note's unsaved draft is kept for the life of the tab, so a
 * reload does not lose it: by workspace, request and item (no item for the
 * request's own notes), and `refKey` (the new note's references, or a
 * saved note's `@id` while it is edited) so two drafts never share one. */
export const draftKey = (
  workspace: string,
  request: string,
  item?: string | null,
  refKey = '',
) =>
  `dw-request-note:${workspace}/${request}/${item ?? ''}` +
  (refKey ? `|${refKey}` : '')

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
