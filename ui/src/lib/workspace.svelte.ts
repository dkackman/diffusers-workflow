import { api, setApiWorkspace } from './api'

const STORAGE_KEY = 'dw-workspace'
export const DEFAULT_WORKSPACE = 'default'

/** Which workspace the UI is looking at, and what the server offers.
 *
 * One store rather than a prop through every page: the selection scopes
 * every request (see api.ts), so a page only has to read `current` inside
 * its load effect to refetch when the selection changes. `names` stays
 * undefined until a listing lands, so "not loaded yet" is distinguishable
 * from "only the default exists". */
export const workspace = $state<{
  current: string
  names: string[] | undefined
  root: string | null
}>({ current: DEFAULT_WORKSPACE, names: undefined, root: null })

/** Restore the last selection before the first request goes out, so a
 * reload lands back in the workspace the user was working in. */
export function restoreWorkspace(): void {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      workspace.current = stored
      setApiWorkspace(stored)
    }
  } catch {
    /* private mode, or storage disabled - the default is a fine answer */
  }
}

export async function loadWorkspaces(): Promise<void> {
  try {
    const result = await api.listWorkspaces()
    workspace.names = result.workspaces.map((entry) => entry.name)
    workspace.root = result.workspace_root
    // A workspace that has gone away (deleted elsewhere, or a stale
    // selection restored from storage) would scope every request to a 404
    if (!workspace.names.includes(workspace.current)) {
      selectWorkspace(DEFAULT_WORKSPACE)
    }
  } catch {
    workspace.names = [DEFAULT_WORKSPACE]
  }
}

export function selectWorkspace(name: string): void {
  workspace.current = name
  setApiWorkspace(name)
  try {
    if (name === DEFAULT_WORKSPACE) localStorage.removeItem(STORAGE_KEY)
    else localStorage.setItem(STORAGE_KEY, name)
  } catch {
    /* the selection still holds for this session */
  }
}
