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

window.addEventListener('hashchange', () => {
  const next = parse()
  route.parts = next.parts
  route.view = next.view
})

export function go(...parts: string[]) {
  location.hash = '/' + parts.map(encodeURIComponent).join('/')
}

/** Navigate to a section of the workspace the UI is in. */
export function goWs(section: WsSection, ...rest: string[]) {
  go('ws', workspace.current, section, ...rest)
}
