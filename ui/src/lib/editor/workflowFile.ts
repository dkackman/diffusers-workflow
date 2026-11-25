// Reading a workflow file into the editor and writing it back: thin
// callers of the editor's load state and the shared saveDocument, so the
// page holds no load or save rules of its own.
import { api } from '../api'
import { ApiError } from '../apiErrors'
import { saveDocument, type SaveOutcome } from '../documentSave'
import {
  NEW_FOLDER,
  type DocumentEditor,
  type RemoteRead,
} from '../editorShell.svelte'
import { writableRoot } from '../libraries'
import type { WorkflowDefinition, WorkflowList } from '../types'

/** Read `name` into `ed` at the ETag and origin it was found with. True
 * once it is loaded; false when the read failed (the error is left on
 * `ed`) or `stale()` says the page has moved on to another file. */
export async function openWorkflow(
  ed: DocumentEditor<WorkflowDefinition>,
  name: string,
  stale: () => boolean = () => false,
): Promise<boolean> {
  ed.beginLoad()
  try {
    const { definition, etag, writable } = await api.getWorkflow(name)
    if (stale()) return false
    ed.load(definition, { etag, path: name, writable })
    return true
  } catch (error) {
    if (!stale()) ed.failLoad(error)
    return false
  }
}

/** `read()`, or `{ gone: true }` when the file it reads is not there */
export async function goneOn404<T extends object>(
  read: () => Promise<RemoteRead<T>>,
): Promise<RemoteRead<T>> {
  try {
    return await read()
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return { gone: true }
    throw error
  }
}

/** `ed`, following the server's copy of the workflow it has open: an
 * agent's or another tab's write reloads it, or raises the banner over
 * unsaved edits, or says it was deleted (DocumentEditor.watchRemote). Call
 * during component init */
export function watchWorkflow(
  ed: DocumentEditor<WorkflowDefinition>,
): DocumentEditor<WorkflowDefinition> {
  ed.watchRemote({
    kind: 'workflows',
    read: () =>
      goneOn404(async () => {
        const read = await api.getWorkflow(ed.loadedPath ?? '')
        const doc = read.definition
        return { doc, etag: read.etag, writable: read.writable }
      }),
  })
  return ed
}

/** Save the document on screen to `path`, asking before it replaces a
 * file it did not read or one that changed since. */
export function saveWorkflowAt(
  ed: DocumentEditor<WorkflowDefinition>,
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

/** The editor's Save, once `validate` passes: the path written, or null
 * when the check failed or a conflict question was cancelled, so nothing
 * was written. The page's action wraps it (busy guard, failure copy) */
export async function validatedSave(
  ed: DocumentEditor<WorkflowDefinition>,
  path: string,
  validate: () => Promise<boolean>,
): Promise<string | null> {
  if (!(await validate())) return null
  ed.busy = true
  try {
    if ((await saveWorkflowAt(ed, path)) === 'cancelled') return null
    ed.commitNewFolder()
    return path
  } finally {
    ed.busy = false
  }
}

/** The editor's Run, once `validate` passes: the document on screen queued
 * as a job, or null when the check failed. base_dir anchors relative paths
 * (images, sub-workflow files) the way running the saved file would - at
 * the workflow's own folder under `root` */
export async function validatedRun(
  ed: DocumentEditor<WorkflowDefinition>,
  root: string,
  validate: () => Promise<boolean>,
) {
  if (!(await validate())) return null
  ed.busy = true
  try {
    const folder = ed.folder && ed.folder !== NEW_FOLDER ? `/${ed.folder}` : ''
    return await api.submitJob({
      workflow: JSON.parse(ed.serialized()) as WorkflowDefinition,
      base_dir: `${root}${folder}`,
    })
  } finally {
    ed.busy = false
  }
}

/** Whether a sub-workflow step's `path` names a workflow file the editor
 * can read: a literal `.json` path, not a reference */
export function subWorkflowPath(path: unknown): path is string {
  return (
    typeof path === 'string' && path.endsWith('.json') && !path.includes(':')
  )
}

/** The variables the sub-workflow at `path` declares, each with its default
 * as a hint. The engine resolves a step's relative path against the PARENT
 * workflow's directory, `folder`, so this read does the same */
export async function workflowVariables(
  path: string,
  folder: string,
): Promise<Array<{ name: string; hint?: string }>> {
  const resolved = folder && !path.startsWith('/') ? `${folder}/${path}` : path
  const fetched = await api.getWorkflow(resolved.slice(0, -'.json'.length))
  return Object.entries(fetched.definition.variables ?? {}).map(
    ([name, value]) => ({
      name,
      hint: typeof value === 'string' ? value : JSON.stringify(value),
    }),
  )
}

/** The definition a new workflow starts from: an example's, or the blank
 * starter for none. Read on the click that chooses it */
export async function startingWorkflow(
  example: string,
  blank: () => WorkflowDefinition,
): Promise<WorkflowDefinition> {
  if (!example) return blank()
  return (await api.getWorkflow(example)).definition
}

/** What the editor takes from the workflow listing: the files the folder
 * picker offers (as `.json` paths), the directory a new file saves under,
 * and the read-only examples a new workflow can start from */
export function listedWorkflows(listing: WorkflowList | undefined) {
  const names = listing?.workflows ?? []
  return {
    files: names.map((file) => `${file}.json`),
    dir: listing ? writableRoot(listing.libraries) : '',
    examples: names.filter((wf) => listing?.details?.[wf]?.writable === false),
  }
}

/** `listing` with `path` in its `key` list: what a page's own save read
 * back, so a folder the save just created is in the picker before the feed
 * says so */
export function withSavedEntry<K extends string, L extends Record<K, string[]>>(
  listing: L,
  key: K,
  path: string,
): L {
  return listing[key].includes(path)
    ? listing
    : { ...listing, [key]: [...listing[key], path] }
}

/** The workflow listing with `path` in it (withSavedEntry) */
export function withSaved(listing: WorkflowList, path: string): WorkflowList {
  return withSavedEntry(listing, 'workflows', path)
}
