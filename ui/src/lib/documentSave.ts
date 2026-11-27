// The one save path the editors share: flush pending edits, refuse a
// document whose load failed, write under the editor's preconditions, and
// turn a 409 (or 412) write conflict into one question - Replace or Cancel - rather
// than a toast - titled "was deleted" when the file the editor read is
// gone. A Replace writes again with no precondition. A saved
// document is a loaded one - a new document's first save included - so
// watchRemote follows the server's copy of it from then on.
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

type Question = { message: string; title?: string }

/** The question a write conflict asks, or null for any other failure. A
 * stale_write carrying no current etag is a file deleted since the read */
function conflictQuestion(error: unknown, label: string): Question | null {
  // A stale If-Match is 409 on the document routes, 412 on the newer ones
  // (the preferences store)
  if (!(error instanceof ApiError) || ![409, 412].includes(error.status))
    return null
  if (error.code === 'stale_write' && error.detail?.etag === null)
    return {
      message: `${label} was deleted since you opened it, in another tab or by an agent. Write yours back?`,
      title: `${label} was deleted`,
    }
  const ask = error.code ? CONFLICTS[error.code] : undefined
  return ask ? { message: ask(label) } : null
}

async function replaceOnConflict(
  error: unknown,
  label: string,
): Promise<boolean> {
  const question = conflictQuestion(error, label)
  if (!question) throw error
  const choices = [
    { label: 'Replace', value: 'replace', danger: true },
    { label: 'Cancel', value: 'cancel' },
  ]
  const answer = question.title
    ? await chooseDialog(question.message, choices, { title: question.title })
    : await chooseDialog(question.message, choices)
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
  ed.loadState = 'loaded'
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

/** The 409 for a name already there, which carries `code: 'exists'`
 * (dw/server/routes/assets.py); other 409s are not that */
export function isExists(error: unknown): boolean {
  return (
    error instanceof ApiError && error.status === 409 && error.code === 'exists'
  )
}

// How many names Keep both tries before giving up
const KEEP_BOTH_TRIES = 20

/** Keep both: `tryName` under the first name `taken` lacks, and the next
 * one while the server says that is there too (the listing was behind).
 * `name` itself is never tried. Each name tried joins `taken`, so a batch
 * does not hand the same one out twice. Throws the last refusal after
 * KEEP_BOTH_TRIES, and any other failure at once */
export async function keepBoth<T>(
  tryName: (name: string) => Promise<T>,
  name: string,
  taken: Set<string>,
): Promise<T> {
  taken.add(name)
  for (let tries = 1; ; tries++) {
    const free = freeName(name, taken)
    try {
      const done = await tryName(free)
      taken.add(free)
      return done
    } catch (error) {
      if (!isExists(error)) throw error
      taken.add(free)
      if (tries >= KEEP_BOTH_TRIES) throw error
    }
  }
}
