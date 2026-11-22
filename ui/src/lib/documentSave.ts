// The one save path the editors share: flush pending edits, refuse a
// document whose load failed, write under the editor's preconditions, and
// turn a 409 write conflict into one question - Replace or Cancel - rather
// than a toast. A Replace writes again with no precondition.
import { ApiError } from './apiErrors'
import type { WritePreconditions } from './apiRequest'
import { chooseDialog } from './confirm.svelte'
import type { DocumentEditor } from './editorShell.svelte'
import { notify } from './toast'

export type SaveOutcome = 'saved' | 'cancelled'

type Write = (pre: WritePreconditions) => Promise<{ etag: string | null }>

const CONFLICTS: Record<string, (label: string) => string> = {
  exists: (label) => `${label} already exists. Replace it?`,
  stale_write: (label) =>
    `${label} changed since you opened it, in another tab or by an agent. Replace that version with yours?`,
}

/** The question a write conflict asks, or null for any other failure. */
function conflictQuestion(error: unknown, label: string): string | null {
  if (!(error instanceof ApiError) || error.status !== 409) return null
  const ask = error.code ? CONFLICTS[error.code] : undefined
  return ask ? ask(label) : null
}

async function replaceOnConflict(
  error: unknown,
  label: string,
): Promise<boolean> {
  const question = conflictQuestion(error, label)
  if (!question) throw error
  const answer = await chooseDialog(question, [
    { label: 'Replace', value: 'replace', danger: true },
    { label: 'Cancel', value: 'cancel' },
  ])
  // Cancel, Escape and a click outside all answer null or 'cancel'
  return answer === 'replace'
}

export async function saveDocument(
  ed: DocumentEditor,
  opts: { path: string; label: string; write: Write },
): Promise<SaveOutcome> {
  ed.flush()
  // Broken JSON on screen: the last valid document is not what the user sees
  if (ed.jsonParseFailed) {
    notify.error('Fix the JSON before saving', 'json-parse')
    return 'cancelled'
  }
  if (!ed.canWrite) throw new Error('not loaded')
  // The baseline is what was sent, not what the document holds once the
  // write returns: an edit made meanwhile must stay dirty
  let sent = ed.serialized()
  let result: { etag: string | null }
  try {
    result = await opts.write(ed.preconditions(opts.path))
  } catch (error) {
    if (!(await replaceOnConflict(error, opts.label))) return 'cancelled'
    sent = ed.serialized()
    result = await opts.write({})
  }
  ed.markSaved(sent)
  ed.etag = result.etag
  ed.loadedPath = opts.path
  // Saved to the workspace: the next save overwrites this copy in place
  ed.writable = true
  return 'saved'
}

/** `name`, or the first `stem-N.ext` (N from 2) that `taken` lacks. */
export function freeName(name: string, taken: ReadonlySet<string>): string {
  if (!taken.has(name)) return name
  const dot = name.lastIndexOf('.')
  const split = dot > name.lastIndexOf('/') + 1 ? dot : name.length
  const stem = name.slice(0, split)
  const ext = name.slice(split)
  let n = 2
  while (taken.has(`${stem}-${n}${ext}`)) n++
  return `${stem}-${n}${ext}`
}
