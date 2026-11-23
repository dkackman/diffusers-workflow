import { api } from './api'
import { ApiError, describeError } from './apiErrors'
import { confirmDialog } from './confirm.svelte'
import { workspaceConfirm } from './destructive'
import { wsHref } from './routes'
import { notify } from './toast'
import {
  DEFAULT_WORKSPACE,
  invalidateWorkspaces,
  loadWorkspaces,
  rememberWorkspace,
  workspace,
} from './workspace.svelte'
import { MAX_NAME_LENGTH, isNameSegment } from './names'

/** The names a workspace cannot take, in any letter case: the folders and
 * files the root itself holds (dw/workspace.py's RESERVED). Checked here so
 * the sidebar can say why before a round trip, but the server has the last
 * word. */
export const RESERVED_WORKSPACE_NAMES = [
  'workflows',
  'prompts',
  'assets',
  'outputs',
  'exports',
  'common',
  'loras',
  'requests.json',
  'workspace.json',
] as const

export const MAX_WORKSPACE_NAME_LENGTH = MAX_NAME_LENGTH

/** What two workspace names are compared by, as dw/security.py's
 * workspace_name_key does: a case-insensitive filesystem holds `Outputs`
 * and `outputs` as one entry. Upper then lower stands in for Python's
 * casefold (ß becomes ss). */
export function workspaceNameKey(name: string): string {
  return name.normalize('NFKC').toUpperCase().toLowerCase()
}

export function workspaceNameError(name: string): string | null {
  if (!name) return 'A workspace needs a name'
  const key = workspaceNameKey(name)
  const reserved = RESERVED_WORKSPACE_NAMES.some(
    (taken) => workspaceNameKey(taken) === key,
  )
  if (reserved) return `${name} is reserved`
  if ([...name].length > MAX_WORKSPACE_NAME_LENGTH)
    return `Use at most ${MAX_WORKSPACE_NAME_LENGTH} characters`
  if (!isNameSegment(name))
    return 'Start with a letter, digit or _; then letters, digits, _, - and .'
  return null
}

/** Create a workspace and land on its overview. Answers the message to show
 * on failure, null on success. */
export async function createWorkspaceAndGo(
  name: string,
): Promise<string | null> {
  const invalid = workspaceNameError(name)
  if (invalid) return invalid
  try {
    await api.createWorkspace(name)
  } catch (e) {
    return describeError(e).title
  }
  invalidateWorkspaces()
  await loadWorkspaces()
  notify.success(`Created workspace ${name}`)
  location.hash = wsHref(name, 'overview')
  return null
}

type Contents = Parameters<typeof workspaceConfirm>[1]

/** What the server's unacknowledged refusal says the workspace holds: the
 * `contents` of its payload, never parsed out of the message */
function contentsOf(e: unknown): Contents {
  const contents = e instanceof ApiError ? e.detail?.contents : undefined
  return contents && typeof contents === 'object' ? (contents as Contents) : {}
}

/** The informed question: the server's counts, and the starred outputs -
 * read off the gallery, since the workspace deleted is the current one.
 * Null (reported) when either could not be read */
async function askToDelete(name: string): Promise<boolean | null> {
  let contents: Contents = {}
  try {
    // Unacknowledged first: the server answers with what it would remove,
    // which is what makes the confirmation an informed one
    await api.deleteWorkspace(name)
  } catch (e) {
    // The server's refusal says what the workspace holds, and is the point
    // of the confirm. Any other failure (offline, a fault) is not something
    // to confirm over: it is reported, with its detail disclosed
    if (describeError(e).kind !== 'refused') {
      notify.failure(`delete ${name}`, e)
      return null
    }
    contents = contentsOf(e)
  }
  let starred: number
  try {
    const { files } = await api.gallery()
    starred = files.filter((f) => f.starred).length
  } catch (e) {
    notify.failure(`count the starred files in ${name}`, e)
    return null
  }
  const spec = workspaceConfirm(name, contents, starred)
  return confirmDialog(spec.message, spec)
}

/** Delete a workspace after an informed confirm - the server's unacknowledged
 * refusal says what it holds - and land on the default overview. */
export async function deleteWorkspaceWithConfirm(
  name: string,
): Promise<boolean> {
  if (name === DEFAULT_WORKSPACE) return false
  if (!(await askToDelete(name))) return false
  try {
    await api.deleteWorkspace(name, true)
  } catch (e) {
    notify.failure(`delete ${name}`, e)
    return false
  }
  // Leave the deleted workspace before the listing reloads: while `current`
  // still names it, the reload would read the route as naming a workspace
  // that does not exist and answer the success with an error toast and a
  // redirect of its own
  workspace.current = DEFAULT_WORKSPACE
  rememberWorkspace(DEFAULT_WORKSPACE)
  invalidateWorkspaces()
  await loadWorkspaces()
  notify.success(`Deleted workspace ${name}`)
  location.hash = wsHref(DEFAULT_WORKSPACE, 'overview')
  return true
}
