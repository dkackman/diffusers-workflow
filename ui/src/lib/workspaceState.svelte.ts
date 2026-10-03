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
