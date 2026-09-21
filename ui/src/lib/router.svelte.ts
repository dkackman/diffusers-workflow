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

function split(): string[] {
  const hash = location.hash.replace(/^#\/?/, '')
  return hash ? hash.split('/').map(decode) : []
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
