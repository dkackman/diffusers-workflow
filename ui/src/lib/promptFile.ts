// Reading a stored prompt into the prompt editor and writing it back: thin
// callers of the editor's load state and the shared saveDocument.
import { api } from './api'
import { saveDocument, type SaveOutcome } from './documentSave'
import type { DocumentEditor } from './editorShell.svelte'
import type { PromptDefinition } from './types'

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
