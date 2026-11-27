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

export const SERVER_SECTIONS = [
  'models',
  'preferences',
  'schema',
  'status',
] as const
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
  server: {
    models: 'Models',
    preferences: 'Preferences',
    schema: 'Schema',
    status: 'Status',
  },
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
  return 'Page not found'
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
  // '#/ws/studio//requests' (a doubled slash) keeps its section: the empty
  // parts after the workspace are dropped before matching
  const [group, second, ...tail] = parts
  const [third, ...rest] = group === 'ws' ? tail.filter(Boolean) : tail
  if (group === 'ws' && second) {
    // No section is the overview; an unknown one is missing, so the URL
    // stays as typed and the page says so
    // A trailing slash ('#/ws/studio/') leaves an empty third part; it
    // reads as no section, so it is the overview too
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

// A comma is left as typed: it is legal in a fragment, and a list reads
// `roles=final,intermediate` rather than `final%2Cintermediate`
const encodeValue = (value: string) =>
  encodeURIComponent(value).replace(/%2C/gi, ',')

/** `values` as a hash's query, empty ones dropped; '' when none is left. */
export function queryString(values: Record<string, string>): string {
  const query = Object.entries(values)
    .filter(([, value]) => value)
    .map(([key, value]) => `${encodeURIComponent(key)}=${encodeValue(value)}`)
    .join('&')
  return query ? '?' + query : ''
}

/** The query keys a section reads that mean the same in any workspace: its
 * filters, never an item's id (`req`, `open`) or one workspace's (`ws`). */
const KEPT_QUERY: Partial<Record<WsSection, string[]>> = {
  workflows: ['q', 'shape', 'traits', 'sort'],
  jobs: ['status', 'q'],
  requests: ['show'],
  gallery: ['q', 'sort', 'group', 'roles', 'starred'],
  assets: ['q', 'sort'],
}

/** Where choosing workspace `name` goes from `view`: the same section of
 * the other workspace, without what follows it (a job id names one job of
 * this workspace). The editor and a comparison name this workspace's files,
 * so they become the other's Workflows and Gallery; a shared or server page
 * belongs to no workspace, so the switch lands on the overview. The filters
 * of `query` that the target section reads go along. */
export function switchHref(
  name: string,
  view: RouteView,
  query: Record<string, string> = {},
): string {
  if (view.kind !== 'ws') return wsHref(name, 'overview')
  const moved: Partial<Record<WsSection, WsSection>> = {
    edit: 'workflows',
    compare: 'gallery',
  }
  const section = moved[view.section] ?? view.section
  const kept = Object.entries(query).filter(([key]) =>
    KEPT_QUERY[section]?.includes(key),
  )
  return wsHref(name, section) + queryString(Object.fromEntries(kept))
}

/** A comparison of `names`: the one builder, each name one path segment
 * (so a name holding a slash survives the round trip through `parseView`,
 * which takes `rest` as the decoded array). */
export function compareHref(
  workspace: string,
  names: string[],
  from?: BackLink,
): string {
  const hash = wsHref(workspace, 'compare', ...names)
  return from
    ? `${hash}?back=${encodeURIComponent(JSON.stringify(from))}`
    : hash
}

/** Where a page was entered from: its back link */
export type BackLink = { href: string; label: string }

/** The `back=` a comparison carries, read back: only an in-app hash link
 * with a label is taken, so a hand-written value cannot point elsewhere. */
export function backFrom(value: string | undefined): BackLink | null {
  if (!value) return null
  try {
    const from = JSON.parse(value) as Partial<BackLink> | null
    return typeof from?.href === 'string' &&
      from.href.startsWith('#/') &&
      typeof from.label === 'string' &&
      from.label
      ? { href: from.href, label: from.label }
      : null
  } catch {
    return null
  }
}

/** A file opened in the Gallery or the Assets viewer: the section's
 * `open=<name>` deep link, which the page opens (and Back closes). */
export function fileHref(
  workspace: string,
  section: 'gallery' | 'assets',
  name: string,
): string {
  return `${wsHref(workspace, section)}?open=${encodeURIComponent(name)}`
}

/** A job's page. With its workspace it is a full hash; without one, only
 * where the caller has none (the header's running link: health carries no
 * workspace), the legacy form the job page redirects. */
export function jobHref(id: string, workspace?: string): string {
  return workspace
    ? wsHref(workspace, 'jobs', id)
    : '#/jobs/' + encodeURIComponent(id)
}
