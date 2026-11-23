// @vitest-environment node
// Which view keeps itself fresh how: every `resource(` call's freshness, as
// written in the source, against the table below. A view of server data is
// a `resource` that names the change kinds it watches (`watch: []` for data
// that never changes), or a `live` poll for machine state, with `alsoLive`
// beside a watch while something is in flight (resource.svelte.ts). A page
// that reads the server with no `resource(` at all fails here by name,
// unless it is a document page, which keeps its document fresh another way.
//
// Outside this table: the app shell's status poll (statusPoll.ts, 5 s, also
// while the tab is hidden), the linked notes' 20 s poll
// (request/listening.svelte.ts), one job's event stream and its resource
// (job/followJob.svelte.ts), and one request's interval, which RequestPage's
// `alsoLive` names (request/refresh.ts).
import { parse } from 'svelte/compiler'
import { expect, it } from 'vitest'

type Freshness =
  | { watch: string[]; alsoLive?: true }
  | { live: { everyMs: number } }
  | { unreadable: string }

/** Each file's resources, in source order, keyed from src/lib */
const PAGE_TABLE: Record<string, Freshness[]> = {
  '../App.svelte': [],
  'Sidebar.svelte': [{ watch: ['workspaces'] }],
  'CachesPanel.svelte': [{ watch: ['models'] }],
  'assets/AssetDetail.svelte': [{ watch: ['assets'] }],
  'editor/LorasEditor.svelte': [{ watch: ['loras'] }],
  'editor/StepEditor.svelte': [{ watch: ['workflows'] }],
  'gallery/GalleryDetail.svelte': [
    { watch: ['gallery'] },
    { watch: ['gallery'] },
  ],
  'gallery/RequestFilter.svelte': [{ watch: ['requests'] }],
  'job/ChainSegments.svelte': [{ watch: ['workflows'] }],
  'job/JobResults.svelte': [{ watch: ['gallery'] }],
  'job/RerunWithChanges.svelte': [{ watch: ['workflows'] }],
  'pages/AssetsPage.svelte': [{ watch: ['assets'] }],
  'pages/ComparePage.svelte': [{ watch: ['gallery'] }],
  'pages/EditorPage.svelte': [{ watch: ['workflows'] }],
  'pages/GalleryPage.svelte': [{ watch: ['gallery'] }],
  'pages/JobPage.svelte': [{ watch: [] }],
  'pages/JobsPage.svelte': [{ live: { everyMs: 3000 } }],
  'pages/JsonEditorStub.svelte': [],
  'pages/ModelsPage.svelte': [
    { watch: ['models'] },
    { watch: ['models'], alsoLive: true },
    { watch: [], alsoLive: true },
  ],
  'pages/OverviewPage.svelte': [
    { watch: ['gallery'] },
    { watch: ['jobs'] },
    { watch: ['workflows'] },
    { watch: ['assets'] },
    { watch: ['requests'] },
    { watch: ['workspaces'] },
  ],
  'pages/PromptEditorPage.svelte': [{ watch: ['prompts'] }],
  'pages/PromptsPage.svelte': [{ watch: ['prompts'] }],
  'pages/RequestPage.svelte': [{ watch: ['requests'], alsoLive: true }],
  'pages/RequestsPage.svelte': [{ watch: ['requests'] }],
  'pages/SchemaPage.svelte': [{ watch: [] }],
  'pages/ServerPage.svelte': [{ watch: [] }],
  'pages/StatusPage.svelte': [],
  'pages/WorkflowPage.svelte': [
    { watch: ['workflows'] },
    { watch: ['gallery'] },
    { watch: ['workflows'] },
  ],
  'pages/WorkflowsPage.svelte': [
    { watch: ['workflows'] },
    { watch: ['gallery'] },
  ],
}

/** The pages that edit one document: it is kept fresh by what they call to
 * open it, not by a resource */
const DOCUMENT_PAGES = [
  'pages/EditorPage.svelte',
  'pages/PromptEditorPage.svelte',
]
const OPENERS: Record<string, string> = {
  'pages/EditorPage.svelte': 'openWorkflow',
  'pages/PromptEditorPage.svelte': 'openPrompt',
}
/** What a document page must call. The opener for now; the remote watch
 * (`watchRemote`) once a document follows the server */
const documentFreshness = (page: string): string => OPENERS[page]

type Node = { type: string; [key: string]: unknown }

function* nodes(node: unknown): Generator<Node> {
  if (Array.isArray(node)) {
    for (const child of node) yield* nodes(child)
    return
  }
  if (!node || typeof node !== 'object') return
  const n = node as Node
  if (typeof n.type === 'string') yield n
  for (const [key, child] of Object.entries(n))
    if (key !== 'parent' && key !== 'loc' && child && typeof child === 'object')
      yield* nodes(child)
}

const calleeName = (call: Node): string | null => {
  const callee = call.callee as Node
  return callee.type === 'Identifier' ? (callee.name as string) : null
}

const property = (object: Node, key: string): Node | undefined =>
  (object.properties as Node[]).find(
    (p) =>
      p.type === 'Property' &&
      ((p.key as Node).name === key || (p.key as Node).value === key),
  )?.value as Node | undefined

/** The freshness a `resource(load, options)` call writes */
function freshnessOf(call: Node, source: string): Freshness {
  const options = (call.arguments as Node[])[1]
  const text = (n: Node) =>
    source.slice(n.start as number, n.end as number).replace(/\s+/g, ' ')
  if (!options || options.type !== 'ObjectExpression')
    return { unreadable: options ? text(options) : '(no options)' }
  const live = property(options, 'live')
  if (live) {
    const every = live.type === 'ObjectExpression' && property(live, 'everyMs')
    return every && every.type === 'Literal'
      ? { live: { everyMs: every.value as number } }
      : { unreadable: text(live) }
  }
  const watch = property(options, 'watch')
  const kinds =
    watch?.type === 'ArrayExpression' ? (watch.elements as Node[]) : null
  if (!kinds || kinds.some((k) => k.type !== 'Literal'))
    return { unreadable: watch ? text(watch) : '(no watch)' }
  const found: Freshness = { watch: kinds.map((k) => k.value as string) }
  if (property(options, 'alsoLive')) found.alsoLive = true
  return found
}

interface Read {
  resources: Freshness[]
  calls: Set<string>
  imports: string[]
}

function read(source: string): Read {
  const ast = parse(source, { modern: true })
  const script = ast.instance?.content
  const calls = [...nodes(script)].filter((n) => n.type === 'CallExpression')
  return {
    resources: calls
      .filter((c) => calleeName(c) === 'resource')
      .map((c) => freshnessOf(c, source)),
    calls: new Set(calls.map(calleeName).filter((n) => n !== null)),
    imports: [...nodes(script)]
      .filter((n) => n.type === 'ImportDeclaration')
      .map((n) => (n.source as Node).value as string),
  }
}

// Every .svelte file's source, keyed from src/lib ('../App.svelte' for the
// shell)
const sources = import.meta.glob<string>(['../App.svelte', './**/*.svelte'], {
  query: '?raw',
  import: 'default',
  eager: true,
})
const key = (path: string) => path.replace(/^\.\//, '')
const isPage = (k: string) => /^pages\/[^/]+\.svelte$/.test(k)
const always = (k: string) =>
  k === '../App.svelte' || k === 'Sidebar.svelte' || isPage(k)

const reads = new Map(
  Object.entries(sources).map(([path, source]) => [key(path), read(source)]),
)

it('every resource declares the freshness the page table names', () => {
  const measured = Object.fromEntries(
    [...reads]
      .filter(([k, r]) => always(k) || r.resources.length > 0)
      .map(([k, r]) => [k, r.resources])
      .sort(([a], [b]) => (a < b ? -1 : 1)),
  )
  expect(measured).toEqual(
    Object.fromEntries(
      Object.entries(PAGE_TABLE).sort(([a], [b]) => (a < b ? -1 : 1)),
    ),
  )
})

it('a page that reads the server does so through a resource', () => {
  const readsServer = (r: Read) =>
    r.imports.some((i) => i === '../api' || /lib\.svelte$/.test(i))
  const bare = [...reads]
    .filter(([k, r]) => isPage(k) && readsServer(r))
    .filter(([k, r]) => r.resources.length === 0 && !DOCUMENT_PAGES.includes(k))
    .map(([k]) => k)
  expect(bare).toEqual([])
})

it('a document page opens its document the way that keeps it fresh', () => {
  const missing = DOCUMENT_PAGES.filter(
    (page) => !reads.get(page)?.calls.has(documentFreshness(page)),
  )
  expect(missing).toEqual([])
})
