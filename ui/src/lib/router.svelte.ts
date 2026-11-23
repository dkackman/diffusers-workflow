import {
  legacyRedirect,
  parseView,
  type RouteView,
  type WsSection,
} from './routes'
import {
  applyRouteWorkspace,
  lastUsedWorkspace,
  setRouteReplacer,
  workspace,
} from './workspace.svelte'

/** Minimal hash router: '#/ws/studio/jobs/abc' -> parts ['ws','studio','jobs','abc']
 * and a typed `view`. A legacy hash is rewritten in place (no history entry)
 * before it is parsed, so '#/gallery' lands in the last-used workspace. */
function decode(part: string): string {
  // A hand-edited or truncated hash can hold a stray '%', which
  // decodeURIComponent throws on - at module load that would blank the whole
  // app, so an undecodable segment is used as it was typed
  try {
    return decodeURIComponent(part)
  } catch {
    return part
  }
}

/** The hash's path and its '?query' tail - '#/ws/studio/jobs?status=failed'.
 * The query is a page's own view state (a list's filters); it is not part
 * of the route, so changing it is not a navigation. */
function hashParts(): [string, string] {
  const hash = location.hash.replace(/^#\/?/, '')
  const at = hash.indexOf('?')
  return at < 0 ? [hash, ''] : [hash.slice(0, at), hash.slice(at + 1)]
}

function split(): string[] {
  const [path] = hashParts()
  return path ? path.split('/').map(decode) : []
}

/** The current hash's query, as key -> value. */
export function hashQuery(): Record<string, string> {
  const query: Record<string, string> = {}
  for (const pair of hashParts()[1].split('&')) {
    if (!pair) continue
    const at = pair.indexOf('=')
    const key = at < 0 ? pair : pair.slice(0, at)
    query[decode(key)] = at < 0 ? '' : decode(pair.slice(at + 1))
  }
  return query
}

/** Replace the hash's query in place: no history entry and no hashchange,
 * so typing in a filter neither floods Back nor re-runs the route. Empty
 * values are dropped, so a cleared filter leaves a clean link. */
export function setHashQuery(values: Record<string, string>) {
  const query = Object.entries(values)
    .filter(([, value]) => value)
    .map(
      ([key, value]) =>
        `${encodeURIComponent(key)}=${encodeURIComponent(value)}`,
    )
    .join('&')
  const hash = '#/' + hashParts()[0] + (query ? '?' + query : '')
  if (hash !== location.hash) history.replaceState(history.state, '', hash)
  committed = location.hash
}

function parse(): { parts: string[]; view: RouteView } {
  let parts = split()
  const redirect = legacyRedirect(parts, lastUsedWorkspace())
  if (redirect) {
    // replaceState raises no second hashchange: the settle that called this
    // (or the module's load) commits the redirected entry
    const hash = '#/' + redirect.map(encodeURIComponent).join('/')
    history.replaceState(history.state, '', hash)
    parts = redirect
  }
  const view = parseView(parts)!
  applyRouteWorkspace(view)
  return { parts, view }
}

export const route = $state(parse())

/** The hash of the route the app is showing. */
let committed = location.hash

type LeaveGuard = () => boolean | Promise<boolean>
// eslint-disable-next-line svelte/prefer-svelte-reactivity -- not reactive state
const guards = new Set<LeaveGuard>()
let asking = false
let undoing = false
// Set by replaceRoute until its hashchange is answered: the arriving entry
// is the committed one renamed, so it keeps the committed index, and a
// decline renames it back rather than walking the history
let replacing = false
// The committed entry's state, as replaceRoute found it
let replacedState: unknown = null

/** Every entry the app commits carries its position in `history.state`, so a
 * declined navigation can tell a pushed link from a Back or Forward
 * traversal and walk the history back to the committed entry. */
function indexOf(): number | undefined {
  const at = (history.state as { dwIndex?: unknown } | null)?.dwIndex
  return typeof at === 'number' ? at : undefined
}

function stamp(at: number) {
  const state = history.state
  const kept = state && typeof state === 'object' ? state : {}
  history.replaceState({ ...kept, dwIndex: at }, '')
}

let committedIndex = indexOf() ?? 0
stamp(committedIndex)
/** The index of the entry the browser was last seen on. */
let lastSeenIndex = committedIndex

/** Stamp the entry a hashchange arrived on, guarded or not: a pushed entry
 * is one past the last one seen, a replaced one keeps its index. So the
 * walk back always knows how far, however many entries arrived while a
 * guard was asking. */
function arrive() {
  const at = indexOf() ?? (replacing ? lastSeenIndex : lastSeenIndex + 1)
  stamp(at)
  lastSeenIndex = at
}

/** Ask before the route leaves its current path (an editor with unsaved
 * edits). Return true to allow. Guards are asked in registration order and
 * never for a query-only change. Returns a function that removes the guard. */
export function addLeaveGuard(guard: LeaveGuard): () => void {
  const entry: LeaveGuard = () => guard()
  guards.add(entry)
  return () => guards.delete(entry)
}

function pathOf(hash: string): string {
  return hash.replace(/^#\/?/, '').split('?')[0]
}

async function allowed(): Promise<boolean> {
  for (const guard of [...guards]) if (!(await guard())) return false
  return true
}

/** Take the entry the browser is on as the route. */
function settle() {
  replacing = false
  replacedState = null
  const at = indexOf() ?? committedIndex + 1
  const next = parse()
  committed = location.hash
  route.parts = next.parts
  route.view = next.view
  committedIndex = at
  lastSeenIndex = at
  stamp(at)
}

/** Back on the committed entry: a replace that renamed it while a guard was
 * asking is renamed back, so the URL matches the route. */
function restore() {
  if (location.hash !== committed)
    history.replaceState(replacedState ?? history.state, '', committed)
  replacing = false
  replacedState = null
  lastSeenIndex = committedIndex
}

/** Walk the history back to the committed entry: the index difference,
 * whether the entries were pushed links or a traversal. The entries left
 * behind keep their own URLs, so Back and Forward still work. A declined
 * replace is no walk at all, only the committed URL put back. */
function undo() {
  const at = indexOf()
  const delta = at === undefined ? -1 : committedIndex - at
  if (!delta) {
    restore()
    return
  }
  undoing = true
  history.go(delta)
}

/** Ask the guards; meanwhile every hashchange only records where the
 * browser is. Allowed, the route settles on that entry, the user's latest
 * choice; declined, the history walks back to the committed one. */
function ask() {
  asking = true
  allowed()
    .catch(() => false)
    .then((ok) => {
      asking = false
      if (ok) settle()
      else undo()
    })
}

window.addEventListener('hashchange', () => {
  if (undoing) {
    // the hashchange of our own history.go
    undoing = false
    restore()
    return
  }
  arrive()
  if (asking) return
  if (!guards.size || pathOf(location.hash) === pathOf(committed)) settle()
  else ask()
})

/** Replace the entry the app is on with `hash`, through the guards - a
 * page correcting its own URL. Declined, the committed URL is put back in
 * place with no history traversal; allowed, the route settles on the same
 * entry, at the same index. */
export function replaceRoute(hash: string) {
  const before = location.hash
  replacing = true
  replacedState = history.state
  location.replace(hash)
  // the same hash raises no hashchange, so nothing will answer the flag
  if (location.hash === before) {
    replacing = false
    replacedState = null
  }
}

/** Rename the entry the app is on - an editor's first save gives its
 * document a path - with no history entry, no hashchange and no guard: the
 * page stays, only what the route calls it changes. `route.parts` is
 * updated in place, so the shell does not take it as a navigation (which
 * scrolls to the top); `route.view` is replaced, which renames the page. */
export function renameRoute(hash: string) {
  history.replaceState(history.state, '', hash)
  committed = location.hash
  const next = parse()
  route.parts.splice(0, route.parts.length, ...next.parts)
  route.view = next.view
}

export function go(...parts: string[]) {
  location.hash = '/' + parts.map(encodeURIComponent).join('/')
}

/** Navigate to a section of the workspace the UI is in. */
export function goWs(section: WsSection, ...rest: string[]) {
  go('ws', workspace.current, section, ...rest)
}

setRouteReplacer(replaceRoute)
