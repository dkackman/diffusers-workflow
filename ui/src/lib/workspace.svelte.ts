import { api } from './api'
import { notify } from './toast'
import { wsHref, type RouteView } from './routes'

const STORAGE_KEY = 'dw-workspace'
export const DEFAULT_WORKSPACE = 'default'

/** Which workspace the UI is looking at, and what the server offers.
 *
 * The route is the source of truth: `applyRouteWorkspace` (called by the
 * router on every change) sets `current` from a `#/ws/<name>/...` hash, and
 * off one (a shared or server page) leaves it at the last workspace a `ws`
 * route named, so a scoped request from those pages still means something.
 * `api.ts` reads `current` directly to scope every request, so a page only
 * has to read `current` inside its load effect to refetch on a switch.
 * `names` stays undefined until a listing lands, so "not loaded yet" is
 * distinguishable from "only the default exists". */
export const workspace = $state<{
  current: string
  names: string[] | undefined
  root: string | null
  /** Roughly how much disk each workspace holds, by name - the server
   * computes it per listing and caches it briefly, so it is a glance
   * rather than a live figure. Absent for a server that does not send it. */
  usage: Record<string, { files: number; bytes: number }>
}>({ current: DEFAULT_WORKSPACE, names: undefined, root: null, usage: {} })

/** The workspace a bare legacy hash ('#/gallery') lands in: the last one a
 * route named, so a reload or an old bookmark goes back where the user was. */
export function lastUsedWorkspace(): string {
  try {
    return localStorage.getItem(STORAGE_KEY) || DEFAULT_WORKSPACE
  } catch {
    return DEFAULT_WORKSPACE
  }
}

export function rememberWorkspace(name: string): void {
  try {
    if (name === DEFAULT_WORKSPACE) localStorage.removeItem(STORAGE_KEY)
    else localStorage.setItem(STORAGE_KEY, name)
  } catch {
    /* the selection still holds for this session */
  }
}

// The view the router last applied, so a listing that lands later can tell
// a workspace the URL named from one the localStorage fallback supplied.
// Recorded here rather than read off the router, which imports this module.
let lastView: RouteView | null = null
// Whether `names` is the server's answer. A failed listing falls back to
// the default alone so the sidebar has something to show, and that fallback
// must not be read as proof that every other workspace is gone.
let listed = false

export function applyRouteWorkspace(view: RouteView | null): void {
  lastView = view
  if (view?.kind === 'ws') {
    workspace.current = view.workspace
    rememberWorkspace(view.workspace)
  } else {
    workspace.current = lastUsedWorkspace()
  }
  checkWorkspaceExists(view)
}

/** A `current` the listing does not hold would scope every request to a
 * 404. Run on every route change and again when the listing lands, so a
 * name typed after the listing is caught as well as one it arrived to. On
 * a `ws` route the URL named the workspace, so it is said out loud and the
 * route replaced; off one (a shared or server page) `current` is only the
 * remembered fallback, which the URL never mentioned - quietly reset, no
 * toast, no navigation. Nothing until a listing has landed, and nothing on
 * the fallback a failed listing leaves. */
export function checkWorkspaceExists(view: RouteView | null): void {
  if (!listed || !workspace.names) return
  if (workspace.names.includes(workspace.current)) return
  const missing = workspace.current
  workspace.current = DEFAULT_WORKSPACE
  rememberWorkspace(DEFAULT_WORKSPACE)
  if (view?.kind === 'ws') {
    notify.error(`No workspace named ${missing} - showing default`)
    location.replace(wsHref(DEFAULT_WORKSPACE, 'overview'))
  }
}

// The in-flight/completed listing, shared across every caller - the sidebar
// and any page that needs names call loadWorkspaces() and would otherwise
// refetch a listing that has not changed.
let workspacesPromise: Promise<void> | null = null

export function loadWorkspaces(): Promise<void> {
  if (!workspacesPromise) {
    workspacesPromise = (async () => {
      try {
        const result = await api.listWorkspaces()
        workspace.names = result.workspaces.map((entry) => entry.name)
        listed = true
        workspace.root = result.workspace_root
        workspace.usage = Object.fromEntries(
          result.workspaces
            .filter((entry) => entry.usage)
            .map((entry) => [entry.name, entry.usage!]),
        )
        // A route naming a workspace that has gone away (deleted elsewhere,
        // or a stale bookmark) is caught here for the route the app opened
        // on; `applyRouteWorkspace` catches every one after
        checkWorkspaceExists(lastView)
      } catch {
        listed = false
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
