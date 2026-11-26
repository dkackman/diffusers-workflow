// A note's draft: when leaving it loses something, and the tab storage it is
// kept in. The storage layer of noteDraft.svelte.ts, which both note editors
// share (NoteEditor and the request's ItemNote).

/** Whether closing the editor would lose words: the draft has text, and it
 * is not the note already saved. */
export function draftAtRisk(draft: string, saved: string | null): boolean {
  return draft.trim() !== '' && draft !== (saved ?? '')
}

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
