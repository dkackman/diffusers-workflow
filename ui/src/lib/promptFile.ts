// Reading a stored prompt into the prompt editor and writing it back: thin
// callers of the editor's load state and the shared saveDocument.
import { api } from './api'
import { saveDocument, type SaveOutcome } from './documentSave'
import type { DocumentEditor } from './editorShell.svelte'
import { goneOn404, withSavedEntry } from './editor/workflowFile'
import { workflowsReferencing } from './prompts'
import type { PromptDefinition, PromptList } from './types'

/** Read `name` into `ed` at the ETag and library it was found in. The
 * prompt once it is loaded; null when the read failed (the error is left
 * on `ed`) or `stale()` says the page has moved on to another prompt. */
export async function openPrompt(
  ed: DocumentEditor<PromptDefinition>,
  name: string,
  stale: () => boolean = () => false,
): Promise<PromptDefinition | null> {
  ed.beginLoad()
  try {
    const { prompt, etag, writable } = await api.getPrompt(name)
    if (stale()) return null
    ed.load(prompt, { etag, path: name, writable })
    return prompt
  } catch (error) {
    if (!stale()) ed.failLoad(error)
    return null
  }
}

/** `ed`, following the server's copy of the prompt it has open: an
 * agent's or another tab's write reloads it, or raises the banner over
 * unsaved edits, or says it was deleted (DocumentEditor.watchRemote). Call
 * during component init */
export function watchPrompt(
  ed: DocumentEditor<PromptDefinition>,
): DocumentEditor<PromptDefinition> {
  ed.watchRemote({
    kind: 'prompts',
    read: () =>
      goneOn404(async () => {
        const read = await api.getPrompt(ed.loadedPath ?? '')
        const doc = read.prompt
        return { doc, etag: read.etag, writable: read.writable }
      }),
  })
  return ed
}

/** Save the prompt on screen to `path`, asking before it replaces a file
 * it did not read or one that changed since. */
export function savePromptAt(
  ed: DocumentEditor<PromptDefinition>,
  path: string,
): Promise<SaveOutcome> {
  return saveDocument(ed, {
    path,
    label: path,
    write: (pre) =>
      api.savePrompt(
        path,
        JSON.parse(ed.serialized()) as PromptDefinition,
        pre,
      ),
  })
}

/** What a delete of `name` is about to break, read on the click that asks:
 * any workflow holding a prompt:name reference fails at run time once the
 * file is gone. '' when none does, or the listing could not be read - the
 * confirm still protects the file itself */
export async function deleteWarning(name: string): Promise<string> {
  try {
    const { details } = await api.listWorkflows()
    const referencing = workflowsReferencing(name, details ?? {})
    if (!referencing.length) return ''
    const shown = referencing.slice(0, 5).join(', ')
    const more = referencing.length > 5 ? ', …' : ''
    return (
      `\n\nReferenced by ${referencing.length} workflow` +
      `${referencing.length === 1 ? '' : 's'} (${shown}${more}) - ` +
      'they will fail at run time until updated.'
    )
  } catch {
    return ''
  }
}

/** The prompt listing with `path` in it (withSavedEntry) */
export function withSavedPrompt(listing: PromptList, path: string): PromptList {
  return withSavedEntry(listing, 'prompts', path)
}
