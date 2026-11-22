// Reading a workflow file into the editor and writing it back: thin
// callers of the editor's load state and the shared saveDocument, so the
// page holds no load or save rules of its own.
import { api } from '../api'
import { saveDocument, type SaveOutcome } from '../documentSave'
import type { DocumentEditor } from '../editorShell.svelte'
import type { WorkflowDefinition } from '../types'

/** Read `name` into `ed` at the ETag and origin it was found with. True
 * once it is loaded; false when the read failed (the error is left on
 * `ed`) or `stale()` says the page has moved on to another file. */
export async function openWorkflow(
  ed: DocumentEditor,
  name: string,
  stale: () => boolean = () => false,
): Promise<boolean> {
  ed.beginLoad()
  try {
    const { definition, etag, writable } = await api.getWorkflow(name)
    if (stale()) return false
    ed.load(definition as Record<string, any>, { etag, path: name, writable })
    return true
  } catch (error) {
    if (!stale()) ed.failLoad(error)
    return false
  }
}

/** Save the document on screen to `path`, asking before it replaces a
 * file it did not read or one that changed since. */
export function saveWorkflowAt(
  ed: DocumentEditor,
  path: string,
): Promise<SaveOutcome> {
  return saveDocument(ed, {
    path,
    label: path,
    write: (pre) =>
      api.saveWorkflow(
        path,
        JSON.parse(ed.serialized()) as WorkflowDefinition,
        pre,
      ),
  })
}
