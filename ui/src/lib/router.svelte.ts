import {
  legacyRedirect,
  parseView,
  type RouteView,
  type WsSection,
} from './routes'
import {
  applyRouteWorkspace,
  lastUsedWorkspace,
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
    location.replace('#/' + redirect.map(encodeURIComponent).join('/'))
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

function apply() {
  const next = parse()
  committed = location.hash
  route.parts = next.parts
  route.view = next.view
}

window.addEventListener('hashchange', () => {
  const target = location.hash
  if (asking || !guards.size || pathOf(target) === pathOf(committed)) {
    // a hashchange during an ask is put back; the ask decides
    if (asking) history.replaceState(history.state, '', committed)
    else apply()
    return
  }
  // hashchange fires after the URL moved: put it back, ask, and on an allow
  // take the target in place (replaceState adds no entry, fires no event)
  history.replaceState(history.state, '', committed)
  asking = true
  allowed()
    .catch(() => false)
    .then((ok) => {
      asking = false
      if (!ok) return
      history.replaceState(history.state, '', target)
      apply()
    })
})

export function go(...parts: string[]) {
  location.hash = '/' + parts.map(encodeURIComponent).join('/')
}

/** Navigate to a section of the workspace the UI is in. */
export function goWs(section: WsSection, ...rest: string[]) {
  go('ws', workspace.current, section, ...rest)
}
