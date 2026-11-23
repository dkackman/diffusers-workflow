// Reading a stored prompt into the prompt editor and writing it back: thin
// callers of the editor's load state and the shared saveDocument.
import { api } from './api'
import { saveDocument, type SaveOutcome } from './documentSave'
import type { DocumentEditor } from './editorShell.svelte'
import { workflowsReferencing } from './prompts'
import type { PromptDefinition, PromptList } from './types'

/** Read `name` into `ed` at the ETag and library it was found in. The
 * prompt once it is loaded; null when the read failed (the error is left
 * on `ed`) or `stale()` says the page has moved on to another prompt. */
export async function openPrompt(
  ed: DocumentEditor,
  name: string,
  stale: () => boolean = () => false,
): Promise<PromptDefinition | null> {
  ed.beginLoad()
  try {
    const { prompt, etag, writable } = await api.getPrompt(name)
    if (stale()) return null
    ed.load(prompt as Record<string, any>, { etag, path: name, writable })
    return prompt
  } catch (error) {
    if (!stale()) ed.failLoad(error)
    return null
  }
}

/** Save the prompt on screen to `path`, asking before it replaces a file
 * it did not read or one that changed since. */
export function savePromptAt(
  ed: DocumentEditor,
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

/** `listing` with `path` in it: what this page's own save read back, so a
 * folder the save just created is in the picker before the feed says so */
export function withSavedPrompt(listing: PromptList, path: string): PromptList {
  return listing.prompts.includes(path)
    ? listing
    : { ...listing, prompts: [...listing.prompts, path] }
}
