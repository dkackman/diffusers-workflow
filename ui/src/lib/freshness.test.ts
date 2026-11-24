// @vitest-environment node
// Which view keeps itself fresh how: every `resource(` call's freshness, as
// written in the source, against the table below. A view of server data is
// a `resource` that names the change kinds it watches (`watch: []` for data
// that never changes), or a `live` poll for machine state, with `alsoLive`
// beside a watch while something is in flight (resource.svelte.ts). A page
// that reads the server with no `resource(` at all fails here by name,
// unless it is a document page, which keeps its document fresh another way:
// it opens it through its opener and watches it with `watchRemote`.
//
// Reads made through a helper: a .svelte file that imports a plain src/lib
// .ts module whose code calls an `api` read (arch-metrics.mjs's `isRead`)
// reads the server too. It must have a `resource(`, be a document page, or
// be in HELPER_EXCEPTIONS with the reason beside it. A file that has a
// `resource(` calls each reading helper inside a resource's load, or the
// call is in HELPER_CALL_EXCEPTIONS by file and name, so a second read
// beside a watched one fails too. Only the script is read: a helper called
// from the markup (an `onclick`) is out of reach. One import level is
// followed: a helper that reaches the api through another module (two or
// more levels) is out of reach here and of the metric, as are `.svelte.ts`
// stores, which keep their own freshness (below).
//
// Outside this table: the app shell's status poll (statusPoll.ts, 5 s, also
// while the tab is hidden), the linked notes' 20 s poll
// (request/listening.svelte.ts), one job's event stream and its resource
// (job/followJob.svelte.ts), and one request's interval, which RequestPage's
// `alsoLive` names (request/refresh.ts).
import { parse } from 'svelte/compiler'
import ts from 'typescript'
import { expect, it } from 'vitest'

// The metric's own read rule. A specifier that is not a literal keeps the
// script out of svelte-check's program, which has no node types
const METRICS = '../../scripts/arch-metrics.mjs'
const { isRead } = (await import(/* @vite-ignore */ METRICS)) as {
  isRead: (name: string) => boolean
}

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
  'editor/ArgumentsEditor.svelte': [{ watch: [] }],
  'editor/LorasEditor.svelte': [{ watch: ['loras'] }],
  'editor/PipelineOptions.svelte': [{ watch: [] }],
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
  'pages/GalleryPage.svelte': [{ watch: ['gallery'] }, { watch: ['gallery'] }],
  'pages/JobPage.svelte': [{ watch: ['jobs'] }],
  'pages/JobsPage.svelte': [{ live: { everyMs: 3000 } }],
  'pages/JsonEditorStub.svelte': [],
  'pages/NotFoundPage.svelte': [],
  'pages/ModelsPage.svelte': [
    { watch: ['models'] },
    { watch: ['models'], alsoLive: true },
    { watch: ['models'], alsoLive: true },
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
  // a text tile's first lines, read once when first seen
  'ui/OutputTile.svelte': [{ watch: [] }],
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
/** What keeps an open document fresh: a `watchRemote` call, in the page or
 * in a helper the page calls (one level, as for reads below) */
const WATCH = 'watchRemote'

/** Files that import a reading helper and have no resource, each with why
 * that is not a view to keep fresh */
const HELPER_EXCEPTIONS: Record<string, string> = {
  '../App.svelte':
    "startStatusPoll is the shell's own 5 s status poll (statusPoll.ts), " +
    'and followJobs reads one job when that poll reports it ended',
  'job/EventLog.svelte':
    'fetchEventLog runs on a click (open, Refresh) and once more when the ' +
    'job ends, to read the events after the last one shown',
  'ui/ConnectionBanner.svelte':
    "checkNow is the Retry now button's one status check",
}

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

const methodName = (call: Node): string | null => {
  const callee = call.callee as Node
  const name = callee.type === 'MemberExpression' && (callee.property as Node)
  return name && name.type === 'Identifier' ? (name.name as string) : null
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
  /** The method names called on an object: `ed.watchRemote(...)` */
  methodCalls: Set<string>
  imports: string[]
  /** Each import's source and the value names it brings in */
  valueImports: { from: string; names: string[] }[]
}

function read(source: string): Read {
  const ast = parse(source, { modern: true })
  const script = ast.instance?.content
  const calls = [...nodes(script)].filter((n) => n.type === 'CallExpression')
  const imports = [...nodes(script)].filter(
    (n) => n.type === 'ImportDeclaration',
  )
  const from = (n: Node) => (n.source as Node).value as string
  return {
    resources: calls
      .filter((c) => calleeName(c) === 'resource')
      .map((c) => freshnessOf(c, source)),
    calls: new Set(calls.map(calleeName).filter((n) => n !== null)),
    methodCalls: new Set(calls.map(methodName).filter((n) => n !== null)),
    imports: imports.map(from),
    valueImports: imports
      .filter((n) => n.importKind !== 'type')
      .map((n) => ({ from: from(n), names: importedNames(n) })),
  }
}

const isApiRead = (n: ts.Node): boolean =>
  ts.isCallExpression(n) &&
  ts.isPropertyAccessExpression(n.expression) &&
  ts.isIdentifier(n.expression.expression) &&
  n.expression.expression.text === 'api' &&
  isRead(n.expression.name.text)

const isWatch = (n: ts.Node): boolean =>
  ts.isCallExpression(n) &&
  ts.isPropertyAccessExpression(n.expression) &&
  n.expression.name.text === WATCH

const hasExport = (n: ts.Node): boolean =>
  ts.canHaveModifiers(n) &&
  (ts.getModifiers(n) ?? []).some((m) => m.kind === ts.SyntaxKind.ExportKeyword)

/** A module's top-level declarations (name to node), and its exports
 * (exported name to local name) */
function declarations(file: ts.SourceFile) {
  const local = new Map<string, ts.Node>()
  const exported = new Map<string, string>()
  for (const statement of file.statements) {
    const named = (name: string, node: ts.Node) => {
      local.set(name, node)
      if (hasExport(statement)) exported.set(name, name)
    }
    if (
      (ts.isFunctionDeclaration(statement) ||
        ts.isClassDeclaration(statement)) &&
      statement.name
    )
      named(statement.name.text, statement)
    else if (ts.isVariableStatement(statement)) {
      for (const d of statement.declarationList.declarations)
        if (ts.isIdentifier(d.name)) named(d.name.text, d)
    } else if (
      ts.isExportDeclaration(statement) &&
      !statement.moduleSpecifier &&
      statement.exportClause &&
      ts.isNamedExports(statement.exportClause)
    )
      for (const e of statement.exportClause.elements)
        exported.set(e.name.text, (e.propertyName ?? e.name).text)
  }
  return { local, exported }
}

/** The names a .ts module exports whose code reaches an `api` read: by
 * calling one, or through another of the module's own declarations that
 * does. A read in a comment is not code, and one made through another
 * module is a second level, out of reach */
function readingExports(
  source: string,
  hit: (n: ts.Node) => boolean = isApiRead,
): Set<string> {
  const file = ts.createSourceFile('m.ts', source, ts.ScriptTarget.Latest)
  const { local, exported } = declarations(file)
  const reads = new Set<string>()
  const uses = new Map<string, Set<string>>()
  for (const [name, node] of local) {
    const used = new Set<string>()
    const visit = (n: ts.Node): void => {
      if (hit(n)) reads.add(name)
      if (ts.isIdentifier(n) && local.has(n.text)) used.add(n.text)
      ts.forEachChild(n, visit)
    }
    visit(node)
    uses.set(name, used)
  }
  for (let grew = true; grew;) {
    grew = false
    for (const [name, used] of uses)
      if (!reads.has(name) && [...used].some((u) => reads.has(u))) {
        reads.add(name)
        grew = true
      }
  }
  return new Set(
    [...exported].filter(([, l]) => reads.has(l)).map(([name]) => name),
  )
}

/** The value names an import declaration brings in ('*' for a namespace) */
const importedNames = (n: Node): string[] =>
  (n.specifiers as Node[])
    .filter((s) => s.importKind !== 'type')
    .map((s) =>
      s.type === 'ImportNamespaceSpecifier'
        ? '*'
        : s.type === 'ImportDefaultSpecifier'
          ? 'default'
          : (((s.imported as Node).name ??
              (s.imported as Node).value) as string),
    )

/** The src/lib-relative key a file's relative import names, if any (the
 * shell's '../App.svelte' climbs out of src/lib, so the walk starts in it) */
function resolved(fromKey: string, specifier: string, known: Set<string>) {
  if (!specifier.startsWith('.')) return null
  const parts = ['src', 'lib']
  const dir = fromKey.split('/').slice(0, -1)
  for (const part of [...dir, ...specifier.split('/')])
    if (part === '..') parts.pop()
    else if (part !== '.') parts.push(part)
  const base = parts.join('/').replace(/^src\/lib\//, '')
  return [base, `${base}.ts`, `${base}/index.ts`].find((k) => known.has(k))
}

/** The .svelte files that import, from a .ts helper, a name whose code
 * calls an `api` read, and have no `resource(` - outside the document pages
 * and `exceptions` */
export function bareHelperReaders(
  svelte: Record<string, string>,
  helpers: Record<string, string>,
  exceptions: Record<string, string> = {},
): string[] {
  const reading = new Map(
    Object.entries(helpers).map(([k, source]) => [k, readingExports(source)]),
  )
  const known = new Set(Object.keys(helpers))
  const readsThrough = (k: string, r: Read) =>
    r.valueImports.some(({ from, names }) => {
      const found = reading.get(resolved(k, from, known) ?? '')
      return (
        !!found &&
        found.size > 0 &&
        names.some((n) => n === '*' || found.has(n))
      )
    })
  return Object.entries(svelte)
    .map(([k, source]) => [k, read(source)] as const)
    .filter(([k, r]) => readsThrough(k, r) && r.resources.length === 0)
    .filter(([k]) => !DOCUMENT_PAGES.includes(k) && !(k in exceptions))
    .map(([k]) => k)
}

type Helpers = { reading: Map<string, Set<string>>; known: Set<string> }

/** The reading helpers a file imports: local name to exported name, and
 * each namespace import's reading names */
function readingLocals(
  k: string,
  script: unknown,
  { reading, known }: Helpers,
) {
  const named = new Map<string, string>()
  const spaces = new Map<string, Set<string>>()
  for (const n of nodes(script)) {
    if (n.type !== 'ImportDeclaration' || n.importKind === 'type') continue
    const found = reading.get(
      resolved(k, (n.source as Node).value as string, known) ?? '',
    )
    if (!found?.size) continue
    for (const spec of n.specifiers as Node[]) {
      const local = (spec.local as Node).name as string
      if (spec.type === 'ImportNamespaceSpecifier') spaces.set(local, found)
      const imported = spec.imported as Node | undefined
      const name = imported && ((imported.name ?? imported.value) as string)
      if (name && spec.importKind !== 'type' && found.has(name))
        named.set(local, name)
    }
  }
  return { named, spaces }
}

/** The reading helper a call names, if it names one */
function helperCalled(
  call: Node,
  { named, spaces }: ReturnType<typeof readingLocals>,
): string | null {
  const callee = call.callee as Node
  if (callee.type === 'Identifier')
    return named.get(callee.name as string) ?? null
  const object = callee.object as Node | undefined
  const prop = callee.property as Node | undefined
  if (callee.type !== 'MemberExpression' || object?.type !== 'Identifier')
    return null
  const found = spaces.get(object.name as string)
  return found?.has(prop?.name as string) ? (prop?.name as string) : null
}

/** The reading helpers a file with a `resource(` calls outside every
 * resource's load: a read nothing keeps fresh, beside one that is */
function strayHelperReads(k: string, source: string, helpers: Helpers) {
  const script = parse(source, { modern: true }).instance?.content
  const calls = [...nodes(script)].filter((n) => n.type === 'CallExpression')
  const loads = calls
    .filter((c) => calleeName(c) === 'resource')
    .map((c) => (c.arguments as Node[])[0])
    .filter(Boolean)
  if (!loads.length) return []
  const locals = readingLocals(k, script, helpers)
  const inLoad = (c: Node) =>
    loads.some(
      (l) =>
        (c.start as number) >= (l.start as number) &&
        (c.end as number) <= (l.end as number),
    )
  const stray = calls
    .filter((c) => !inLoad(c))
    .map((c) => helperCalled(c, locals))
  return [...new Set(stray.filter((n) => n !== null))]
}

/** `file: helper` for each reading helper a file with a resource calls
 * outside every resource's load - outside the document pages, which keep
 * their document fresh another way, and `exceptions` */
export function helperReadsOutsideResources(
  svelte: Record<string, string>,
  helperSources: Record<string, string>,
  exceptions: Record<string, Record<string, string>> = {},
): string[] {
  const helpers: Helpers = {
    reading: new Map(
      Object.entries(helperSources).map(([k, src]) => [k, readingExports(src)]),
    ),
    known: new Set(Object.keys(helperSources)),
  }
  return Object.entries(svelte)
    .filter(([k]) => !DOCUMENT_PAGES.includes(k))
    .flatMap(([k, source]) =>
      strayHelperReads(k, source, helpers)
        .filter((name) => !exceptions[k]?.[name])
        .map((name) => `${k}: ${name}`),
    )
}

/** The document pages whose open document nothing watches: neither the
 * page nor a helper function it imports and calls reaches `watchRemote` */
export function unwatchedDocuments(
  svelte: Record<string, string>,
  helpers: Record<string, string>,
  pages: readonly string[] = DOCUMENT_PAGES,
): string[] {
  const known = new Set(Object.keys(helpers))
  const watches = (k: string, r: Read) =>
    r.methodCalls.has(WATCH) ||
    r.valueImports.some(({ from, names }) => {
      const helper = helpers[resolved(k, from, known) ?? '']
      if (helper === undefined) return false
      const found = readingExports(helper, isWatch)
      return names.some((n) => found.has(n) && r.calls.has(n))
    })
  return pages.filter((k) => {
    const source = svelte[k]
    return source === undefined || !watches(k, read(source))
  })
}

// Every .svelte file's source, keyed from src/lib ('../App.svelte' for the
// shell)
const sources = import.meta.glob<string>(['../App.svelte', './**/*.svelte'], {
  query: '?raw',
  import: 'default',
  eager: true,
})
const key = (path: string) => path.replace(/^\.\//, '')
// Every plain .ts module's source (no tests, no .svelte.ts stores)
const helperSources = Object.fromEntries(
  Object.entries(
    import.meta.glob<string>(
      ['./**/*.ts', '!./**/*.test.ts', '!./**/*.svelte.ts', '!./**/*.d.ts'],
      { query: '?raw', import: 'default', eager: true },
    ),
  ).map(([path, source]) => [key(path), source]),
)
const svelteSources = Object.fromEntries(
  Object.entries(sources).map(([path, source]) => [key(path), source]),
)
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
/** Reading-helper calls outside every resource's load in a file that has
 * one, by file and helper name, each with why it is not a second read */
const HELPER_CALL_EXCEPTIONS: Record<string, Record<string, string>> = {
  'job/JobResults.svelte': {
    runMetadataReader:
      'a factory, called once at init: the reader it returns is what the ' +
      'resource calls',
  },
}

it('a document page opens its document through its opener', () => {
  const missing = DOCUMENT_PAGES.filter(
    (page) => !reads.get(page)?.calls.has(OPENERS[page]),
  )
  expect(missing).toEqual([])
})

it('a document page watches the document it opens', () => {
  expect(unwatchedDocuments(svelteSources, helperSources)).toEqual([])
})

it('names a document page whose open document nothing watches', () => {
  const helpers = {
    'editor/file.ts': `export function watchFile(ed) {
  ed.watchRemote({ kind: 'workflows', read: () => read(ed) })
  return ed
}
// ed.watchRemote() in a comment is not a call
export const openFile = (ed) => ed.load({})`,
  }
  const page = (body: string) => `<script lang="ts">
  import { openFile, watchFile } from '../editor/file'
  const ed = new DocumentEditor({})
  ${body}
</script>`
  const svelte = {
    'pages/Itself.svelte': page(
      "ed.watchRemote({ kind: 'prompts', read })\n  openFile(ed)",
    ),
    'pages/Helper.svelte': page('watchFile(ed)\n  openFile(ed)'),
    'pages/Imported.svelte': page('openFile(ed)'),
    'pages/Opener.svelte': `<script lang="ts">
  import { openFile } from '../editor/file'
  openFile(ed)
</script>`,
  }
  expect(unwatchedDocuments(svelte, helpers, Object.keys(svelte))).toEqual([
    'pages/Imported.svelte',
    'pages/Opener.svelte',
  ])
})

it('a view that reads through a helper does so through a resource', () => {
  expect(
    bareHelperReaders(svelteSources, helperSources, HELPER_EXCEPTIONS),
  ).toEqual([])
})

it('each helper exception still reads through a helper with no resource', () => {
  const stale = Object.keys(HELPER_EXCEPTIONS).filter(
    (k) => !bareHelperReaders(svelteSources, helperSources).includes(k),
  )
  expect(stale).toEqual([])
})

it('names a component that calls a reading helper with no resource', () => {
  const helpers = {
    'job/probe.ts': `import { api } from '../api'
// api.listJobs() in a comment is not a call
export const jobsFor = (w: string) => api.listJobs(w)`,
    'job/quiet.ts': `import { api } from '../api'
export const save = () => api.saveWorkflow('a', {})`,
    'job/indirect.ts': `import { api } from '../api'
export const LIMIT = 4
function fetchOne(id: string) { return api.getJob(id) }
const viaLocal = (id: string) => fetchOne(id)
export { viaLocal as latestJob }`,
  }
  const svelte = {
    'job/Bare.svelte': `<script lang="ts">
  import { jobsFor } from './probe'
  let rows = $state([])
  $effect(() => { jobsFor('w').then((r) => (rows = r)) })
</script>`,
    'job/Watched.svelte': `<script lang="ts">
  import { jobsFor } from './probe'
  const jobs = resource(() => jobsFor('w'), { watch: ['jobs'] })
</script>`,
    'job/Writes.svelte': `<script lang="ts">
  import { save } from './quiet'
</script>`,
    'job/Types.svelte': `<script lang="ts">
  import type { jobsFor } from './probe'
</script>`,
    'job/Limit.svelte': `<script lang="ts">
  import { LIMIT } from './indirect'
</script>`,
    'job/Latest.svelte': `<script lang="ts">
  import { latestJob } from './indirect'
</script>`,
    'pages/Clicks.svelte': `<script lang="ts">
  import { jobsFor } from '../job/probe'
</script>`,
  }
  expect(bareHelperReaders(svelte, helpers)).toEqual([
    'job/Bare.svelte',
    'job/Latest.svelte',
    'pages/Clicks.svelte',
  ])
  expect(
    bareHelperReaders(svelte, helpers, { 'pages/Clicks.svelte': 'a click' }),
  ).toEqual(['job/Bare.svelte', 'job/Latest.svelte'])
  expect(bareHelperReaders(svelte, {})).toEqual([])
})

it('a view with a resource makes its helper reads inside one', () => {
  expect(
    helperReadsOutsideResources(
      svelteSources,
      helperSources,
      HELPER_CALL_EXCEPTIONS,
    ),
  ).toEqual([])
})

it('each helper-call exception is still a call outside a resource', () => {
  const found = helperReadsOutsideResources(svelteSources, helperSources)
  const stale = Object.entries(HELPER_CALL_EXCEPTIONS)
    .flatMap(([k, names]) => Object.keys(names).map((n) => `${k}: ${n}`))
    .filter((entry) => !found.includes(entry))
  expect(stale).toEqual([])
})

it('names a second read beside a watched one', () => {
  const helpers = {
    'job/probe.ts': `import { api } from '../api'
export const jobsFor = (w: string) => api.listJobs(w)
export const jobFor = (id: string) => api.getJob(id)`,
  }
  const watched = `const jobs = resource(() => jobsFor('w'), { watch: ['jobs'] })`
  const svelte = {
    'job/Inside.svelte': `<script lang="ts">
  import { jobsFor } from './probe'
  ${watched}
</script>`,
    'job/Beside.svelte': `<script lang="ts">
  import { jobsFor, jobFor as one } from './probe'
  ${watched}
  let job = $state()
  $effect(() => { one('a').then((j) => (job = j)) })
</script>`,
    'job/Spaced.svelte': `<script lang="ts">
  import * as probe from './probe'
  const jobs = resource(() => probe.jobsFor('w'), { watch: ['jobs'] })
  void probe.jobFor('a')
</script>`,
    'job/Bare.svelte': `<script lang="ts">
  import { jobsFor } from './probe'
  void jobsFor('w')
</script>`,
  }
  expect(helperReadsOutsideResources(svelte, helpers)).toEqual([
    'job/Beside.svelte: jobFor',
    'job/Spaced.svelte: jobFor',
  ])
  expect(
    helperReadsOutsideResources(svelte, helpers, {
      'job/Beside.svelte': { jobFor: 'a click' },
    }),
  ).toEqual(['job/Spaced.svelte: jobFor'])
})
