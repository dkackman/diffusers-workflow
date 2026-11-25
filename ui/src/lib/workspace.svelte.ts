import { api } from './api'
import { notify } from './toast'
import { wsHref, type RouteView } from './routes'
import { DEFAULT_WORKSPACE, workspace } from './workspaceState.svelte'
import { onReconnect } from './connection.svelte'
import { latestOnly } from './latest'

// The state lives in its own module so api.ts can scope requests by it
// without importing this one, which imports api.ts to load the listing
export { DEFAULT_WORKSPACE, workspace }

const STORAGE_KEY = 'dw-workspace'

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
// Whether `names` was fetched for the check now reading it. The listing is
// cached for the page's life, and a workspace can be created elsewhere after
// it landed (an agent's create_workspace, another tab) - the running chip's
// job link then corrects the URL into one the cache has never heard of. So
// a name the cache lacks is refetched once before it is called missing.
let fresh = false
let refetching = false
// How a missing workspace's route is replaced: the router hands over its
// replaceRoute at load, since it imports this module and so cannot be imported
let replaceRoute: (hash: string) => void = () => {}

export function setRouteReplacer(replace: (hash: string) => void): void {
  replaceRoute = replace
}

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
 * route replaced - `current` follows only if that replace lands: a declined
 * one keeps the editor on screen, and its requests must keep naming the
 * missing workspace (a 404) rather than write into the default one. Off one (a shared or server page) `current` is only the
 * remembered fallback, which the URL never mentioned - quietly reset, no
 * toast, no navigation. Nothing until a listing has landed, and nothing on
 * the fallback a failed listing leaves. */
export function checkWorkspaceExists(view: RouteView | null): void {
  if (!workspace.listed || !workspace.names) return
  if (workspace.names.includes(workspace.current)) return
  if (!fresh) {
    // The refetch's own landing re-runs this check against `lastView`
    if (refetching) return
    refetching = true
    invalidateWorkspaces()
    void loadWorkspaces().finally(() => (refetching = false))
    return
  }
  if (view?.kind === 'ws') {
    notify.error(`No workspace named ${workspace.current} - showing default`)
    // The route that lands sets `current` (applyRouteWorkspace)
    replaceRoute(wsHref(DEFAULT_WORKSPACE, 'overview'))
    return
  }
  workspace.current = DEFAULT_WORKSPACE
  rememberWorkspace(DEFAULT_WORKSPACE)
}

/** The check with `names` counted as fetched for it - a name it lacks is
 * missing, not refetched again. */
function checkAgainstFreshListing(): void {
  fresh = true
  try {
    checkWorkspaceExists(lastView)
  } finally {
    fresh = false
  }
}

// The in-flight/completed listing, shared across every caller - the sidebar
// and any page that needs names call loadWorkspaces() and would otherwise
// refetch a listing that has not changed.
let workspacesPromise: Promise<void> | null = null

export function loadWorkspaces(): Promise<void> {
  if (!workspacesPromise) workspacesPromise = newest = readListing()
  return workspacesPromise
}

// Reads can overlap (an invalidate while one is out) and land in either
// order: only the one started last writes the listing and checks the route
// against it. An overtaken one settles with `newest`, so whoever awaits it
// still finds a listing at least as new as the one they asked for
const begin = latestOnly()
let newest: Promise<void>

async function readListing(): Promise<void> {
  const current = begin()
  try {
    const result = await api.listWorkspaces()
    if (!current()) return newest
    workspace.names = result.workspaces.map((entry) => entry.name)
    workspace.listed = true
    workspace.root = result.workspace_root
    workspace.usage = Object.fromEntries(
      result.workspaces
        .filter((entry) => entry.usage)
        .map((entry) => [entry.name, entry.usage!]),
    )
    workspace.changed = Object.fromEntries(
      result.workspaces
        .filter((entry) => entry.mtime)
        .map((entry) => [entry.name, entry.mtime!]),
    )
    workspace.notes = Object.fromEntries(
      result.workspaces
        .filter((entry) => entry.note)
        .map((entry) => [entry.name, entry.note!]),
    )
    // A route naming a workspace that has gone away (deleted elsewhere,
    // or a stale bookmark) is caught here for the route the app opened
    // on; `applyRouteWorkspace` catches every one after
    checkAgainstFreshListing()
  } catch {
    if (!current()) return newest
    // The first listing failing leaves the fallback. A refetch failing
    // (the one `checkWorkspaceExists` runs for a name the cache lacks)
    // keeps the listing that landed: a server hiccup does not empty the
    // sidebar, and the name that listing lacks is still called missing,
    // since that listing is the only answer there is.
    if (workspace.listed) checkAgainstFreshListing()
    else workspace.names = [DEFAULT_WORKSPACE]
  }
}

/** Drop the cached listing so the next `loadWorkspaces()` refetches - call
 * after a create or delete changes what the server offers. */
export function invalidateWorkspaces(): void {
  workspacesPromise = null
}

/** The listing read again, the cached one dropped first: what a view
 * watching the `workspaces` kind loads. Never a read already on its way,
 * which may have left before the change it was asked for */
export function relistWorkspaces(): Promise<void> {
  invalidateWorkspaces()
  return loadWorkspaces()
}

// A listing read while the server was down or locked is the default-only
// fallback; once it lets this browser in again, read the real one
onReconnect(() => {
  invalidateWorkspaces()
  void loadWorkspaces()
})
