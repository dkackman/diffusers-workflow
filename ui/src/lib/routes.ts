/** The shape of the hash: '#/ws/<name>/<section>/...', '#/shared/<section>/...',
 * '#/server/<section>'. Pure - the router applies it. Anything else is a
 * legacy hash (the pre-workspace '#/gallery' and friends) and is mapped to
 * its new home by `legacyRedirect`, so an old bookmark still lands. */

export const WS_SECTIONS = [
  'overview',
  'workflows',
  'jobs',
  'requests',
  'gallery',
  'compare',
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

/** The one set of section names: the sidebar's items and the breadcrumb
 * both read it, so they cannot drift. */
export const SECTION_LABELS: {
  ws: Record<WsSection, string>
  shared: Record<SharedSection, string>
  server: Record<ServerSection, string>
  groups: { shared: string; server: string }
} = {
  ws: {
    overview: 'Overview',
    workflows: 'Workflows',
    jobs: 'Jobs',
    requests: 'Requests',
    gallery: 'Gallery',
    compare: 'Compare',
    assets: 'Assets',
    edit: 'Editor',
  },
  shared: {
    prompts: 'Prompts',
    'prompt-edit': 'Prompt editor',
    assets: 'Assets',
    examples: 'Examples',
  },
  server: { models: 'Models', schema: 'Schema', status: 'Status' },
  groups: { shared: 'Shared', server: 'Server' },
}

export type RouteView =
  | { kind: 'ws'; workspace: string; section: WsSection; rest: string[] }
  | { kind: 'shared'; section: SharedSection; rest: string[] }
  | { kind: 'server'; section: ServerSection }
  | { kind: 'missing'; path: string }

/** The name a view goes by in the breadcrumb. */
export function sectionLabel(view: RouteView): string {
  if (view.kind === 'ws') return SECTION_LABELS.ws[view.section]
  if (view.kind === 'shared') return SECTION_LABELS.shared[view.section]
  if (view.kind === 'server') return SECTION_LABELS.server[view.section]
  return 'Not found'
}

const missing = (parts: string[]): RouteView => ({
  kind: 'missing',
  path: '/' + parts.join('/'),
})

/** What the router shows for a hash: its view, or `missing` for one no
 * route (and no legacy redirect) names. */
export function resolveView(parts: string[]): RouteView {
  return parseView(parts) ?? missing(parts)
}

const isWs = (s: string): s is WsSection =>
  (WS_SECTIONS as readonly string[]).includes(s)
const isShared = (s: string): s is SharedSection =>
  (SHARED_SECTIONS as readonly string[]).includes(s)
const isServer = (s: string): s is ServerSection =>
  (SERVER_SECTIONS as readonly string[]).includes(s)

export function parseView(parts: string[]): RouteView | null {
  const [group, second, third, ...rest] = parts
  if (group === 'ws' && second) {
    // No section is the overview; an unknown one is missing, so the URL
    // stays as typed and the page says so
    if (third === undefined) {
      return { kind: 'ws', workspace: second, section: 'overview', rest: [] }
    }
    if (isWs(third))
      return { kind: 'ws', workspace: second, section: third, rest }
    return missing(parts)
  }
  if (group === 'shared' && second) {
    if (!isShared(second)) return missing(parts)
    return {
      kind: 'shared',
      section: second,
      rest: third ? [third, ...rest] : [],
    }
  }
  if (group === 'server' && second) {
    return isServer(second)
      ? { kind: 'server', section: second }
      : missing(parts)
  }
  return null
}

/** Where a hash that `parseView` refuses should go. `lastUsed` is the
 * workspace the scoped legacy routes land in. Null when nothing needs
 * redirecting, an unknown path included (it is a missing route). */
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
      return null
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
