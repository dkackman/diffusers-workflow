/** The shape of the hash: '#/ws/<name>/<section>/...', '#/shared/<section>/...',
 * '#/server/<section>'. Pure - the router applies it. Anything else is a
 * legacy hash (the pre-workspace '#/gallery' and friends) and is mapped to
 * its new home by `legacyRedirect`, so an old bookmark still lands. */

export const WS_SECTIONS = [
  'overview',
  'workflows',
  'jobs',
  'gallery',
  'assets',
  'edit',
] as const
export type WsSection = (typeof WS_SECTIONS)[number]

export const SHARED_SECTIONS = [
  'prompts',
  'prompt-edit',
  'assets',
  'examples',
] as const
export type SharedSection = (typeof SHARED_SECTIONS)[number]

export const SERVER_SECTIONS = ['models', 'schema', 'status'] as const
export type ServerSection = (typeof SERVER_SECTIONS)[number]

export type RouteView =
  | { kind: 'ws'; workspace: string; section: WsSection; rest: string[] }
  | { kind: 'shared'; section: SharedSection; rest: string[] }
  | { kind: 'server'; section: ServerSection }

const isWs = (s: string): s is WsSection =>
  (WS_SECTIONS as readonly string[]).includes(s)
const isShared = (s: string): s is SharedSection =>
  (SHARED_SECTIONS as readonly string[]).includes(s)
const isServer = (s: string): s is ServerSection =>
  (SERVER_SECTIONS as readonly string[]).includes(s)

export function parseView(parts: string[]): RouteView | null {
  const [group, second, third, ...rest] = parts
  if (group === 'ws' && second) {
    // An unknown section is the overview rather than a 404: the workspace
    // is the thing the link names, and the overview is where it starts
    if (third === undefined) {
      return { kind: 'ws', workspace: second, section: 'overview', rest: [] }
    }
    if (isWs(third))
      return { kind: 'ws', workspace: second, section: third, rest }
    return { kind: 'ws', workspace: second, section: 'overview', rest: [] }
  }
  if (group === 'shared' && second && isShared(second)) {
    return {
      kind: 'shared',
      section: second,
      rest: third ? [third, ...rest] : [],
    }
  }
  if (group === 'server' && second && isServer(second)) {
    return { kind: 'server', section: second }
  }
  return null
}

/** Where a hash that `parseView` refuses should go. `lastUsed` is the
 * workspace the scoped legacy routes land in. Null when nothing needs
 * redirecting. */
export function legacyRedirect(
  parts: string[],
  lastUsed: string,
): string[] | null {
  if (parseView(parts)) return null
  const [top, ...rest] = parts
  const ws = (section: WsSection, ...tail: string[]) => [
    'ws',
    lastUsed,
    section,
    ...tail,
  ]
  switch (top) {
    case undefined:
    case '':
    case 'ws':
      return ws('overview')
    case 'workflows':
      return ws('workflows', ...rest)
    case 'gallery':
      return ws('gallery')
    case 'assets':
      return ws('assets')
    case 'edit':
      return ws('edit', ...rest)
    case 'jobs':
      // The queue spans every workspace; one job belongs to one. The job's
      // own workspace is not known until it loads, so the page corrects the
      // URL then (JobPage) - lastUsed is the first guess
      return rest.length ? ws('jobs', ...rest) : ['server', 'status']
    case 'prompts':
      return ['shared', 'prompts']
    case 'prompt-edit':
      return ['shared', 'prompt-edit', ...rest]
    case 'shared':
      return ['shared', 'prompts']
    case 'models':
      return ['server', 'models']
    case 'schema':
      return ['server', 'schema']
    case 'server':
      return ['server', 'status']
    default:
      return ws('overview')
  }
}

const join = (parts: string[]) => '#/' + parts.map(encodeURIComponent).join('/')

export function wsHref(
  workspace: string,
  section: WsSection,
  ...rest: string[]
): string {
  return join(['ws', workspace, section, ...rest])
}
export function sharedHref(section: SharedSection, ...rest: string[]): string {
  return join(['shared', section, ...rest])
}
export function serverHref(section: ServerSection): string {
  return join(['server', section])
}
