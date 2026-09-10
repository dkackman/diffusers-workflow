import { api } from './api'

const STORAGE_KEY = 'dw-workspace'
export const DEFAULT_WORKSPACE = 'default'

/** Which workspace the UI is looking at, and what the server offers.
 *
 * One store rather than a prop through every page: `api.ts` reads
 * `workspace.current` directly to scope every request, so a page only has
 * to read `current` inside its load effect to refetch when the selection
 * changes. `names` stays undefined until a listing lands, so "not loaded
 * yet" is distinguishable from "only the default exists". */
export const workspace = $state<{
  current: string
  names: string[] | undefined
  root: string | null
  /** Roughly how much disk each workspace holds, by name - the server
   * computes it per listing and caches it briefly, so it is a glance
   * rather than a live figure. Absent for a server that does not send it. */
  usage: Record<string, { files: number; bytes: number }>
}>({ current: DEFAULT_WORKSPACE, names: undefined, root: null, usage: {} })

/** Restore the last selection before the first request goes out, so a
 * reload lands back in the workspace the user was working in. */
export function restoreWorkspace(): void {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) workspace.current = stored
  } catch {
    /* private mode, or storage disabled - the default is a fine answer */
  }
}

// The in-flight/completed listing, shared across every caller - a page
// switch calls loadWorkspaces() again (WorkspacePicker mounts fresh each
// time) and would otherwise refetch a listing that has not changed.
let workspacesPromise: Promise<void> | null = null

export function loadWorkspaces(): Promise<void> {
  if (!workspacesPromise) {
    workspacesPromise = (async () => {
      try {
        const result = await api.listWorkspaces()
        workspace.names = result.workspaces.map((entry) => entry.name)
        workspace.root = result.workspace_root
        workspace.usage = Object.fromEntries(
          result.workspaces
            .filter((entry) => entry.usage)
            .map((entry) => [entry.name, entry.usage!]),
        )
        // A workspace that has gone away (deleted elsewhere, or a stale
        // selection restored from storage) would scope every request to a 404
        if (!workspace.names.includes(workspace.current)) {
          selectWorkspace(DEFAULT_WORKSPACE)
        }
      } catch {
        workspace.names = [DEFAULT_WORKSPACE]
      }
    })()
  }
  return workspacesPromise
}

/** Drop the cached listing so the next `loadWorkspaces()` refetches - call
 * after a create or delete changes what the server offers. */
export function invalidateWorkspaces(): void {
  workspacesPromise = null
}

export function selectWorkspace(name: string): void {
  workspace.current = name
  try {
    if (name === DEFAULT_WORKSPACE) localStorage.removeItem(STORAGE_KEY)
    else localStorage.setItem(STORAGE_KEY, name)
  } catch {
    /* the selection still holds for this session */
  }
}
