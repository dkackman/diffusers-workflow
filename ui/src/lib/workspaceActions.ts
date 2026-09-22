import { api } from './api'
import { confirmDialog } from './confirm.svelte'
import { wsHref } from './routes'
import { notify } from './toast'
import {
  DEFAULT_WORKSPACE,
  invalidateWorkspaces,
  loadWorkspaces,
  rememberWorkspace,
  workspace,
} from './workspace.svelte'

/** The names a workspace cannot take: the folders the root itself holds
 * (dw/workspace.py's RESERVED). Checked here so the sidebar can say why
 * before a round trip, but the server has the last word. */
export const RESERVED_WORKSPACE_NAMES = [
  'workflows',
  'prompts',
  'assets',
  'outputs',
  'exports',
  'common',
] as const

const NAME = /^[A-Za-z0-9_][A-Za-z0-9_-]*$/

export function workspaceNameError(name: string): string | null {
  if (!name) return 'A workspace needs a name'
  if ((RESERVED_WORKSPACE_NAMES as readonly string[]).includes(name))
    return `${name} is reserved`
  if (!NAME.test(name)) return 'Use letters, digits, - and _ only'
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
    return e instanceof Error ? e.message : String(e)
  }
  invalidateWorkspaces()
  await loadWorkspaces()
  notify.success(`Created workspace ${name}`)
  location.hash = wsHref(name, 'overview')
  return null
}

/** Delete a workspace after an informed confirm - the server's unacknowledged
 * refusal says what it holds - and land on the default overview. */
export async function deleteWorkspaceWithConfirm(
  name: string,
): Promise<boolean> {
  if (name === DEFAULT_WORKSPACE) return false
  let detail = ''
  try {
    // Unacknowledged first: the server answers with what it would remove,
    // which is what makes the confirmation an informed one
    await api.deleteWorkspace(name)
  } catch (e) {
    detail = e instanceof Error ? e.message : String(e)
  }
  if (
    !(await confirmDialog(`${detail}\n\nDelete workspace "${name}"?`.trim(), {
      confirmLabel: 'Delete',
    }))
  )
    return false
  try {
    await api.deleteWorkspace(name, true)
  } catch (e) {
    notify.error(e instanceof Error ? e.message : String(e))
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
