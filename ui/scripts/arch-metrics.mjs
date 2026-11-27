// Architecture metrics for ui/src - the UI's half of the stabilization
// ratchet (scripts/arch_metrics.py measures dw and dw_mcp). Every metric is
// lower-is-better, so a ratchet is one comparison against the committed
// baseline:
//
//   node scripts/arch-metrics.mjs                  print the metrics as JSON
//   node scripts/arch-metrics.mjs --write PATH     record a baseline
//   node scripts/arch-metrics.mjs --check PATH     exit 1 on any regression
//   node scripts/arch-metrics.mjs --compare CUR BASE   the regressions
//                                                  between two measurements
//
// Two kinds of metric. An enforced one rising is a regression: printed on
// stdout as `key: base -> cur` (the line harnest's check_ui parses) and
// --check/--compare exit 1. An ADVISORY one (files_over_size_ceiling,
// complex_functions, long_functions, duplicate_blocks - proxies for shape,
// not broken rules) rising is a warning: printed on stderr as
// `warning (advisory): key: base -> cur`, then the items its detail names,
// and the exit code does not change. Both kinds are recorded by --write.
//
// Counting rules, fixed so any commit measures the same way:
// - Sources are ui/src's .ts, .svelte and .css files, tests excluded
//   (*.test.ts, *.svelte.test.ts).
// - files_over_size_ceiling (advisory): files above SIZE_CEILING raw
//   lines; a .svelte file's <style> counts, since it is part of what a
//   reader holds. Detail: `file (N lines)`.
// - files_over_hard_ceiling (enforced): files above HARD_SIZE_CEILING raw
//   lines, counted the same way - the outer limit a file cannot grow past.
// - complex_functions (advisory): ESLint's core `complexity` rule above
//   COMPLEXITY_LIMIT, script blocks only - ESLint does not see template
//   {#if}/{#each} branching. Detail: `file:line name (complexity)`.
// - long_functions (advisory): ESLint's `max-lines-per-function` above
//   FUNCTION_LINE_LIMIT, blank and comment lines skipped. Detail:
//   `file:line name (lines)`.
// - import_cycles / modules_in_import_cycles: strongly connected components
//   of more than one module over relative imports, type-only imports
//   included (they still couple the files).
// - layering_violations: relative import declarations from one folder to
//   another that the layer rules forbid, type-only imports included, one per
//   declaration. src/lib/ui/ and the top-level src/lib modules (the shared
//   layer) import only the shared layer; a feature folder imports itself, the
//   shared layer and what LAYER_ALLOWS lists for it; pages/ (TOP_LAYER) and
//   src/App.svelte, src/main.ts import anything. A folder under src/lib in
//   none of LAYER_ALLOWS, SHARED_LAYER and TOP_LAYER counts every edge out
//   of it, and arch-metrics.test.ts fails naming it. Detail:
//   `file:line from-folder -> to-folder`, the top-level modules as `lib`.
// - prefix_literals: string literals and template parts that start with a
//   reference prefix, outside PREFIX_OWNERS. The prefixes are read from
//   dw/references.py, the engine's owner.
// - a11y_suppressions: `svelte-ignore a11y_*` comments.
// - raw_error_toasts: `notify.error(` calls whose first argument reads
//   `.message` or `String(x)` - an error shown raw, not described. An
//   identifier argument is followed to its const (or single-assignment let)
//   initializer.
// - empty_outside_loadstate: imports of `*/Empty.svelte` anywhere but
//   LOAD_STATE; a page states an empty list through LoadState.
// - unwatched_reads: in a .svelte file or a .svelte.ts store,
//   `api.<read>(...)` calls with no `resource(` or `poll(` call around them,
//   less a store's STORE_EXCEPTIONS. In a store `this.api` and `this.#api`
//   count as `api`. A read is `get*`, `list*` or a
//   name in READ_ALIASES. Every other `api` key is in NOT_READS, and
//   arch-metrics.test.ts fails, naming the key, when one is in neither list
//   or in both - so a new method cannot dodge this count unclassified.
// Known limits of those three rules:
// - `resource` and `poll` are matched by callee name.
// - `api` is matched by identifier name, so an alias or destructuring is
//   invisible.
// - a read made through a plain `.ts` helper is not counted here.
//   src/lib/freshness.test.ts sees one import level: a .svelte file that
//   imports a src/lib .ts module calling an `api` read must have a
//   `resource(` or be a declared exception. A read two or more imports
//   away is out of reach of both checks.
// - an `Empty` re-exported through a barrel is not counted.
// - `String(...)` matches a single identifier only.
// - window_dialogs: calls to `confirm`, `prompt` or `alert`, bare or on
//   `window`, `globalThis` or `self` (`window.confirm`, `self['alert']`) -
//   a dialog is `confirmDialog`/`chooseDialog` (src/lib/confirm.svelte.ts).
// - unguarded_mutations: in a .svelte.ts store, an `api.<write>` call or
//   value (WRITES) inside no `action(` call's arguments, less its
//   STORE_EXCEPTIONS; and in a .svelte file, an `api.<write>` call (WRITES),
//   an `api.<write>` handed on as a value, or a use of an imported plain .ts
//   function whose code calls one, lexically inside no `action(` call's
//   arguments and inside no guarded function declaration. `action` is the
//   one imported from a module named `action.svelte` (the name is matched,
//   not resolved to src/lib), under whatever local name; a local of that
//   name guards nothing. A function declaration is guarded
//   when it is used and every use of it outside its own body (a scope
//   reference, so neither a property key nor a shadowing local) is inside an
//   `action(` call's arguments: a helper also bound to `onclick={helper}` is
//   not. A .ts function writes when its code calls an `api` write directly
//   or through another of its module's own top-level declarations - one
//   import level, as src/lib/freshness.test.ts follows reads - and api.ts's
//   own EXPORTED_WRITES (`setNote`) write. A write handed as a value to an
//   imported function, or to a method of an instance of an imported class
//   (`picks.download(api.archiveAssets)`, `const picks = new Picks()`), is
//   guarded when that module's function hands the parameter only to
//   `action(`, directly or through a local function that does.
// - banned_terms: user-facing copy matching a BANNED pattern
//   (src/lib/terms.ts), outside COPY_OWNERS. Copy is a string literal (a
//   single word included; a module specifier is not copy), a template
//   literal with its parts joined by "0", Svelte markup text outside
//   <style>, and a static attribute value. Each pattern counts once per
//   piece of copy. Detail is sorted by file, line and term.
// - duplicate_blocks (advisory): over .svelte files, lines trimmed and their
//   whitespace collapsed, blank and TRIVIAL_LINE lines dropped; a 6-line
//   window whose sha1 occurs in two or more files is shared, and every line
//   inside a shared window is counted once - lines, not runs, so a fix in
//   the middle of a run lowers the count and never raises it. Detail:
//   `file: N lines (from-to, ...)`, source line ranges.
// - raw_media: media elements outside RAW_MEDIA_OWNERS (OutputTile and
//   MediaBody), in .svelte, .ts and .css files; a media file is shown
//   through them. Counted forms: `<img>`, `<video>`, `<audio>`, `<source>`,
//   `<picture>`, `<track>`, `<embed>`, `<object>`, `<canvas>` (one held
//   inside another, `<video><source>`, counts with its holder);
//   `<svelte:element this>` whose static or braced value - either side of a
//   conditional - is one of them; `X.createElement('img')` with a static
//   name; `new Audio()` and `new Image()`; `{@html expr}` whose expression
//   text, or a same-file variable's initializer and assignments, holds one
//   of those tags; and a CSS `background-image: url(` (or `background:` with
//   a `url(`). Detail: `file:line form`, sorted.
// Known limits of those four rules:
// - a write two or more imports away (a .ts helper that writes through
//   another module) is not counted. A use of a `.svelte.ts` store is not a
//   write either (their exports are classes and factories whose methods
//   write); the write is counted, or excepted, in the store.
// - a store's `api` handed to it under another name than `api` is invisible.
// - only `function` declarations can be guarded helpers; a write inside an
//   arrow helper (`const f = () => api.x()`) counts even when an action
//   names it. Make it a `function` declaration that only an action names,
//   or name the write in the action itself.
// - a guarding method is found through `const x = new Class()` only; an
//   instance passed in or returned by a factory is not followed.
// - copy assembled across `{...}` in markup can slip through banned_terms.
import { ESLint } from 'eslint'
import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs'
import {
  dirname,
  join,
  relative,
  resolve,
  extname,
  isAbsolute,
  sep,
} from 'node:path'
import { fileURLToPath } from 'node:url'
import { BANNED, COPY_OWNERS } from '../src/lib/terms.ts'

const UI = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const SRC = join(UI, 'src')
const REFERENCES_PY = join(UI, '..', 'dw', 'references.py')

export const SIZE_CEILING = 600
export const HARD_SIZE_CEILING = 900
export const COMPLEXITY_LIMIT = 10
export const FUNCTION_LINE_LIMIT = 60
export const PREFIX_OWNERS = new Set(['src/lib/references.ts'])

const LIB = join(SRC, 'lib')
/** The shared layer: src/lib/ui/ and the top-level src/lib modules (named
 * `lib` in detail). It imports only itself */
export const SHARED_LAYER = new Set(['ui'])
/** Folders that may import anything, as src/App.svelte and src/main.ts do */
export const TOP_LAYER = new Set(['pages'])
/** Which feature folders each feature folder may import, besides itself
 * and the shared layer. A folder under src/lib in none of these three lists
 * counts every edge out of it, and arch-metrics.test.ts fails naming it */
export const LAYER_ALLOWS = {
  job: ['gallery', 'request', 'workflow', 'editor'],
  gallery: ['request'],
  workflow: ['editor'],
  editor: [],
  assets: [],
  request: [],
  // SchemaNode, read by pages/SchemaPage only
  schema: [],
  // the Server → Preferences key editor, read by pages/PreferencesPage; it
  // shows each tier's pick with the run check's rung list
  ladder: ['editor'],
  // test support (apiMock and its kin) - non-test sources should not import it
  test: [],
}

export const LOAD_STATE = 'src/lib/ui/LoadState.svelte'
export const READ_ALIASES = [
  'gallery',
  'galleryMetadata',
  'health',
  'memory',
  'diffusersStatus',
  'dwStatus',
  'server',
  'eventLog',
  'describeTask',
  'describePipeline',
  'describeClass',
  'restartBatch',
  'fileText',
  'notesChannel',
  'workspaceChanges',
]
/** The `api` keys that write: each moves server state, and the change feed
 * reports it */
export const WRITES = [
  'saveWorkflow',
  'savePrompt',
  'rerunJob',
  'rerunRestartBatch',
  'moveJob',
  'exportJob',
  'cancelJob',
  'submitJob',
  'startDownload',
  'cancelDownload',
  'deleteModel',
  'updateDiffusers',
  'updateDw',
  'clearCache',
  'deleteCacheEntry',
  'deleteOutput',
  'setStarred',
  'archiveOutputs',
  'uploadMedia',
  'archiveAssets',
  'deleteAsset',
  'putRequest',
  'deleteRequest',
  'addNote',
  'editNote',
  'deleteNote',
  'createWorkspace',
  'deleteWorkspace',
  'keepOutput',
  'deleteWorkflow',
  'deletePrompt',
  'enhance',
  'setPreference',
  'resetPreference',
  'setDefaultTier',
]
/** api.ts's exports outside the `api` object that write: a use of one is a
 * write, as an `api.<write>` call is */
export const EXPORTED_WRITES = ['setNote']
/** api.ts's other exports outside `api`; arch-metrics.test.ts fails, naming
 * the export, when one is in neither list */
export const EXPORTED_NOT_WRITES = [
  'ApiError',
  'errorDetail',
  'outputUrl',
  'fetchOutputText',
  'TERMINAL_STATUSES',
  'streamJobEvents',
]
/** Every `api` key that is not a read, by what it is instead */
export const NOT_READS = [
  ...WRITES,
  // checks a click asks for: each answers one question about what the user
  // holds now, a snapshot nothing renders as a view to keep fresh
  'validate',
  'assess',
  'checkToken',
  // a precondition read for a save, not a view: the note and etag a save
  // sends as If-Match
  'noteForSave',
  // URL builders: no request is made
  'exportZipUrl',
  'outputDownloadUrl',
  'workflowDownloadUrl',
  'promptDownloadUrl',
]

/** The only components that may hold a raw media element */
export const RAW_MEDIA_OWNERS = new Set([
  'src/lib/ui/OutputTile.svelte',
  'src/lib/ui/MediaBody.svelte',
])
const MEDIA_TAGS = new Set([
  'img',
  'video',
  'audio',
  'source',
  'picture',
  'track',
  'embed',
  'object',
  'canvas',
])
const MEDIA_HTML =
  /<\s*(img|video|audio|source|picture|track|embed|object|canvas)\b/i
const MEDIA_CONSTRUCTORS = new Set(['Audio', 'Image'])
/** A CSS background that paints a picture: `background-image: url(`, the
 * `background:` shorthand holding a `url(`, or the Svelte directive
 * `style:background-image={...}` (`style:background=` is a colour as often
 * as a picture, so it is not counted) */
const MEDIA_CSS =
  /background(?:-image)?\s*:[^;{}<>]*\burl\(|style:background-image\s*=/g

/** The static text of a string literal or an expression-free template */
function staticText(node) {
  if (node?.type === 'Literal' && typeof node.value === 'string')
    return node.value
  if (node?.type === 'TemplateLiteral' && node.expressions.length === 0)
    return node.quasis.map((q) => q.value.cooked ?? q.value.raw).join('')
  return null
}

/** Every static string an expression can be: a literal, or either side of a
 * conditional or a logical expression */
function staticValues(node) {
  if (node?.type === 'ConditionalExpression')
    return [...staticValues(node.consequent), ...staticValues(node.alternate)]
  if (node?.type === 'LogicalExpression')
    return [...staticValues(node.left), ...staticValues(node.right)]
  const text = staticText(node)
  return text === null ? [] : [text]
}

/** The variable an identifier names, through enclosing scopes */
function variableOf(sourceCode, id) {
  for (let s = sourceCode.getScope(id); s; s = s.upper) {
    const v = s.set.get(id.name)
    if (v) return v
  }
  return null
}

/** The media tag a `<svelte:element this=...>` can be, if any: a static
 * `this="img"`, or a braced one (`this={...}`, a special directive) whose
 * literal - either branch of a conditional - names one */
function elementTag(node) {
  const values = []
  for (const a of node.startTag?.attributes ?? []) {
    if (a.type === 'SvelteSpecialDirective' && a.kind === 'this')
      values.push(...staticValues(a.expression))
    else if (a.type === 'SvelteAttribute' && a.key?.name === 'this')
      values.push(...a.value.map((v) => v.value))
  }
  return values.find((t) => MEDIA_TAGS.has(t)) ?? null
}

/** The media tag an element is, if it is one */
const mediaTag = (node) => {
  const tag = node.name?.name
  if (MEDIA_TAGS.has(tag)) return tag
  return tag === 'svelte:element' ? elementTag(node) : null
}

/** The source an `{@html}` shows: its expression, and an identifier's
 * initializer and every value assigned to it in the same file */
function htmlText(sourceCode, expression) {
  const texts = [sourceCode.getText(expression)]
  if (expression.type !== 'Identifier') return texts.join('\n')
  for (const ref of variableOf(sourceCode, expression)?.references ?? [])
    if (ref.writeExpr) texts.push(sourceCode.getText(ref.writeExpr))
  return texts.join('\n')
}

/** Media elements outside RAW_MEDIA_OWNERS, however they are written: markup
 * (a nested `<source>` or `<img>` counted with the element that holds it),
 * `<svelte:element this>`, `document.createElement('video')`, `new Audio()` /
 * `new Image()`, and an `{@html}` string holding a media tag */
function mediaRule(found) {
  return {
    create(context) {
      if (isMediaOwner(context.filename)) return {}
      const sourceCode = context.sourceCode
      const add = (node, what) =>
        found.push(
          `${relative(UI, context.filename)}:${node.loc.start.line} ${what}`,
        )
      return {
        SvelteElement(node) {
          const tag = mediaTag(node)
          if (!tag) return
          const held = sourceCode
            .getAncestors(node)
            .some((a) => a.type === 'SvelteElement' && mediaTag(a))
          if (!held) add(node, tag)
        },
        SvelteMustacheTag(node) {
          if (
            node.kind === 'raw' &&
            MEDIA_HTML.test(htmlText(sourceCode, node.expression))
          )
            add(node, 'html')
        },
        CallExpression(node) {
          const c = node.callee
          if (c.type !== 'MemberExpression' || c.computed) return
          if (c.property.name !== 'createElement') return
          const tag = staticText(node.arguments[0])?.toLowerCase()
          if (MEDIA_TAGS.has(tag)) add(node, tag)
        },
        NewExpression(node) {
          if (
            node.callee.type === 'Identifier' &&
            MEDIA_CONSTRUCTORS.has(node.callee.name)
          )
            add(node, `new ${node.callee.name}`)
        },
      }
    },
  }
}

const isMediaOwner = (file) =>
  RAW_MEDIA_OWNERS.has(relative(UI, file).replaceAll('\\', '/'))

/** CSS backgrounds that paint a picture, outside RAW_MEDIA_OWNERS. The whole
 * text of a .css or .svelte file is scanned, not only <style> and style
 * attributes, so one in a script string or markup text counts too; a
 * background set from a script (`el.style.background = ...`) is not seen */
export function cssMedia(file, text) {
  if (isMediaOwner(file) || file.endsWith('.ts')) return []
  return [...text.matchAll(MEDIA_CSS)].map((m) => {
    const line = text.slice(0, m.index).split('\n').length
    return `${relative(UI, file)}:${line} css background`
  })
}

export const isLoadState = (file) =>
  relative(UI, file).replaceAll('\\', '/') === LOAD_STATE

const RAW_ERROR = /\.message\b|String\(\s*\w+\s*\)/
// The text a toast argument stands for: an identifier is followed to its
// `const` (or single-assignment `let`) initializer, through enclosing scopes.
function isRawError(sourceCode, call, arg) {
  if (RAW_ERROR.test(sourceCode.getText(arg))) return true
  if (arg.type !== 'Identifier') return false
  const init = fixedValue(variableOf(sourceCode, arg))
  return Boolean(init) && RAW_ERROR.test(sourceCode.getText(init))
}

/** A variable's one value: a `const`'s initializer, or a `let` assigned
 * only by its initializer */
function fixedValue(v) {
  const def = v?.defs.length === 1 ? v.defs[0] : null
  if (def?.type !== 'Variable' || !def.node.init) return null
  const writes = v.references.filter((r) => r.isWrite()).length
  return def.parent?.kind === 'const' || writes === 1 ? def.node.init : null
}
export const isRead = (name) =>
  /^(get|list)/.test(name) || READ_ALIASES.includes(name)

/** What is wrong with how `keys` (the `api` object's) are classified: a key
 * that is neither a read nor in NOT_READS, or both, and a list entry no key
 * has */
export function misclassified(
  keys,
  aliases = READ_ALIASES,
  notReads = NOT_READS,
) {
  const read = (k) => /^(get|list)/.test(k) || aliases.includes(k)
  const problems = []
  for (const k of keys) {
    if (read(k) && notReads.includes(k))
      problems.push(`${k}: both a read and in NOT_READS`)
    else if (!read(k) && !notReads.includes(k))
      problems.push(`${k}: neither a read nor in NOT_READS`)
  }
  for (const k of [...aliases, ...notReads])
    if (!keys.includes(k))
      problems.push(`${k}: listed, but api has no such key`)
  return problems
}
const isMember = (node, object, property) =>
  node?.type === 'MemberExpression' &&
  node.object.type === 'Identifier' &&
  node.object.name === object &&
  (property === null || node.property.name === property)

const SOURCE_SUFFIXES = new Set(['.ts', '.svelte', '.css'])

export function sourceFiles(root = SRC) {
  const found = []
  for (const name of readdirSync(root)) {
    const path = join(root, name)
    if (statSync(path).isDirectory()) {
      // Generated from the server's OpenAPI document - measured on the server side
      if (name !== 'generated') found.push(...sourceFiles(path))
    } else if (SOURCE_SUFFIXES.has(extname(name)) && !/\.test\.ts$/.test(name))
      found.push(path)
  }
  return found.sort()
}

export function referencePrefixes(text = readFileSync(REFERENCES_PY, 'utf8')) {
  const prefixes = []
  for (const match of text.matchAll(/^[A-Z_]+ = "([a-z_]+:)"$/gm))
    prefixes.push(match[1])
  if (prefixes.length === 0)
    throw new Error('no reference prefixes found in dw/references.py')
  return prefixes
}

/** Tarjan's strongly connected components, over {module: [module]} */
export function cycles(graph) {
  let index = 0
  const stack = []
  const onStack = new Set()
  const indices = new Map()
  const lowlinks = new Map()
  const components = []
  const visit = (node) => {
    indices.set(node, index)
    lowlinks.set(node, index)
    index++
    stack.push(node)
    onStack.add(node)
    for (const next of graph[node] ?? []) {
      if (!indices.has(next)) {
        visit(next)
        lowlinks.set(node, Math.min(lowlinks.get(node), lowlinks.get(next)))
      } else if (onStack.has(next)) {
        lowlinks.set(node, Math.min(lowlinks.get(node), indices.get(next)))
      }
    }
    if (lowlinks.get(node) === indices.get(node)) {
      const component = []
      let member
      do {
        member = stack.pop()
        onStack.delete(member)
        component.push(member)
      } while (member !== node)
      if (component.length > 1) components.push(component.sort())
    }
  }
  for (const node of Object.keys(graph)) if (!indices.has(node)) visit(node)
  return components
}

/** A file's layer: its folder under `lib`, `lib` for a top-level module
 * there, or `src` for anything outside it (App.svelte, main.ts) */
function layerOf(file, lib) {
  const rel = relative(lib, file)
  if (rel.startsWith('..') || isAbsolute(rel)) return 'src'
  const parts = rel.split(sep)
  return parts.length === 1 ? 'lib' : parts[0]
}

const isShared = (layer) => layer === 'lib' || SHARED_LAYER.has(layer)

function layerAllows(from, to) {
  if (from === to || from === 'src' || TOP_LAYER.has(from)) return true
  if (isShared(from)) return isShared(to)
  if (!Object.hasOwn(LAYER_ALLOWS, from)) return false
  return isShared(to) || LAYER_ALLOWS[from].includes(to)
}

/** Import edges `{ from, line, target }` that cross folders against the
 * layer rules, as `file:line from-folder -> to-folder`, sorted */
export function layeringViolations(edges, lib = LIB) {
  const found = []
  for (const { from, line, target } of edges) {
    const a = layerOf(from, lib)
    const b = layerOf(target, lib)
    if (!layerAllows(a, b))
      found.push(`${relative(UI, from)}:${line} ${a} -> ${b}`)
  }
  return found.sort(byPlace)
}

/** Folders under `lib` in none of LAYER_ALLOWS, SHARED_LAYER, TOP_LAYER */
export function undeclaredFolders(lib = LIB) {
  return readdirSync(lib)
    .filter(
      (name) =>
        name !== 'generated' &&
        statSync(join(lib, name)).isDirectory() &&
        !Object.hasOwn(LAYER_ALLOWS, name) &&
        !SHARED_LAYER.has(name) &&
        !TOP_LAYER.has(name),
    )
    .sort()
}

function resolveImport(from, specifier, known) {
  if (!specifier.startsWith('.')) return null
  const base = resolve(dirname(from), specifier)
  for (const candidate of [base, `${base}.ts`, join(base, 'index.ts')])
    if (known.has(candidate)) return candidate
  return null
}

const DIALOG_NAMES = new Set(['alert', 'confirm', 'prompt'])
const DIALOG_HOSTS = new Set(['window', 'globalThis', 'self'])
/** The name a member reads: `a.b`, or a literal `a['b']` */
const propertyName = (member) =>
  member.computed ? staticText(member.property) : member.property.name
const isWindowDialog = (callee) =>
  (callee.type === 'Identifier' && DIALOG_NAMES.has(callee.name)) ||
  (callee.type === 'MemberExpression' &&
    callee.object.type === 'Identifier' &&
    DIALOG_HOSTS.has(callee.object.name) &&
    DIALOG_NAMES.has(propertyName(callee)))

/** The module `action` comes from: matched by name only (any module named
 * `action.svelte`), the specifier is not resolved to src/lib/action.svelte.ts */
const ACTION_MODULE = /(^|\/)action\.svelte(\.[jt]s)?$/
/** A call of the `action` that action.svelte.ts exports, under whatever
 * local name it was imported as; a local of that name is no guard */
function isActionCall(node, sourceCode) {
  if (node.type !== 'CallExpression' || node.callee.type !== 'Identifier')
    return false
  const def = variableOf(sourceCode, node.callee)?.defs[0]
  return (
    def?.type === 'ImportBinding' &&
    def.node.type === 'ImportSpecifier' &&
    importedName(def.node) === 'action' &&
    ACTION_MODULE.test(def.parent.source.value)
  )
}

/** The reads and writes a `.svelte.ts` store makes outside `resource(`,
 * `poll(` and `action(`, each with why that is right. Keyed
 * `<file> <declaration> <api name>` - the outermost function, method or
 * top-level const holding it - so another one in the same file is counted.
 * arch-metrics.test.ts fails on an entry nothing matches any more. */
export const STORE_EXCEPTIONS = {
  reads: {
    'src/lib/assetlib.svelte.ts assets listAssets':
      'read-once picker cache (lazyListing): per workspace, reloaded after an upload; nothing watches it',
    'src/lib/outputlib.svelte.ts outputs gallery':
      'read-once picker cache (lazyListing): per workspace, reloaded when MediaPicker opens on it again; nothing watches it',
    'src/lib/promptlib.svelte.ts loadPromptLibrary listPrompts':
      'read when a page that offers prompts loads; nothing watches it',
    'src/lib/editorLists.svelte.ts loadEditorLists listPipelines':
      'class and task names of the installed engine, read when the editor loads',
    'src/lib/editorLists.svelte.ts loadEditorLists listClasses':
      'class and task names of the installed engine, read when the editor loads',
    'src/lib/editorLists.svelte.ts loadEditorLists listTasks':
      'class and task names of the installed engine, read when the editor loads',
    'src/lib/editorLists.svelte.ts loadEditorLists listBuiltinWorkflows':
      'packaged fragments of the installed engine, read when the editor loads',
    'src/lib/changes.svelte.ts ChangeFeed.refresh workspaceChanges':
      'the change feed resource() subscribes to; refresh() runs under its own poll()',
    'src/lib/enhancer.svelte.ts Enhancer.load listEnhancers':
      'presets of the installed engine, read when the enhance panel opens',
    'src/lib/enhancer.svelte.ts Enhancer.refreshModels listModels':
      'the cached-model chip: read when the panel opens and after a download',
    'src/lib/enhancer.svelte.ts Enhancer.downloadModel listDownloads':
      'the loop following a download this panel started, until it ends or stop()',
    'src/lib/enhancer.svelte.ts Enhancer.#follow getJob':
      'read only when the job event stream drops, to tell whether the job ended',
    'src/lib/enhancer.svelte.ts Enhancer.#finish getJob':
      'the one read of a finished job, for its result',
    'src/lib/request/linkedRequest.svelte.ts LinkedRequest.read getRequest':
      'read when the job page or gallery box opens a link; its note box then rereads it in a poll() and after each change',
    'src/lib/request/listening.svelte.ts NotesChannel.refresh notesChannel':
      'run by the poll() in hold() while a note list is on screen, and before a save',
    'src/lib/request/planStore.svelte.ts requestRoutes getRequest':
      "PlanStore's get: RequestPage's resource loads through PlanStore.refresh",
    'src/lib/workspace.svelte.ts readListing listWorkspaces':
      'behind loadWorkspaces(), which Sidebar and Overview run in a resource watching `workspaces`',
  },
  writes: {
    'src/lib/enhancer.svelte.ts Enhancer.downloadModel startDownload':
      'the Download button gives way to a "downloading" chip as soon as it is pressed',
    'src/lib/enhancer.svelte.ts Enhancer.generate enhance':
      'Generate is disabled while `busy`, set before the call; a failure is shown in the panel',
    'src/lib/enhancer.svelte.ts Enhancer.cancel cancelJob':
      'cancelling twice is harmless, and a refusal means the job had already ended',
    'src/lib/request/planStore.svelte.ts requestRoutes putRequest':
      "PlanStore's put: one write at a time from its queue, each failure shown to the page",
    'src/lib/request/runQueue.svelte.ts RunQueue.#send submitJob':
      'RunQueue sends nothing while `busy`, set before the first call; each row shows its failure',
  },
}

/** The `api` key `node` reaches in a store: `api.x`, or `x` on the `api` a
 * class holds (`this.api.x`, `this.#api.x`) */
function storeApiName(node) {
  if (node.type !== 'MemberExpression' || node.computed) return null
  const on = node.object
  const held =
    (on.type === 'Identifier' && on.name === 'api') ||
    (on.type === 'MemberExpression' &&
      on.object.type === 'ThisExpression' &&
      on.property.name === 'api')
  return held ? node.property.name : null
}

/** The outermost named declaration holding `node`: `fn`, `Class.method` or
 * a top-level const */
function declarationOf(node, sourceCode) {
  let cls = ''
  for (const a of sourceCode.getAncestors(node)) {
    if (a.type === 'ClassDeclaration') cls = `${a.id.name}.`
    if (a.type === 'MethodDefinition' || a.type === 'PropertyDefinition') {
      const hash = a.key.type === 'PrivateIdentifier' ? '#' : ''
      return `${cls}${hash}${a.key.name}`
    }
    if (a.type === 'FunctionDeclaration' || a.type === 'VariableDeclarator')
      return a.id.name
  }
  return '(module)'
}

/** A store's site: its exception key, and the detail line it counts as */
function storeSite(node, name, context) {
  const file = relative(UI, context.filename).replaceAll('\\', '/')
  return {
    key: `${file} ${declarationOf(node, context.sourceCode)} ${name}`,
    site: `${file}:${node.loc.start.line} ${name}`,
  }
}

const isStore = (file) => file.endsWith('.svelte.ts')

/** Every `api` read in a .svelte file with no `resource(` or `poll(` around
 * it goes into `found`; a store's into `stores`, for its exceptions */
function readsRule(found, stores) {
  return {
    create(context) {
      const store = isStore(context.filename)
      if (!store && !context.filename.endsWith('.svelte')) return {}
      return {
        CallExpression(node) {
          const callee = node.callee
          const name = store
            ? storeApiName(callee)
            : isMember(callee, 'api', null) && callee.property.name
          if (!name || !isRead(name)) return
          const watched = context.sourceCode
            .getAncestors(node)
            .some(
              (a) =>
                a.type === 'CallExpression' &&
                a.callee.type === 'Identifier' &&
                (a.callee.name === 'resource' || a.callee.name === 'poll'),
            )
          if (watched) return
          if (store) stores.push(storeSite(node, name, context))
          else found.push(relative(UI, context.filename))
        },
      }
    },
  }
}

/** Every `api` write a store calls or hands on that no `action(` call's
 * arguments hold, into `stores` */
function storeWritesRule(stores) {
  return {
    create(context) {
      if (!isStore(context.filename)) return {}
      const writes = []
      const actionRanges = []
      return {
        CallExpression(node) {
          if (isActionCall(node, context.sourceCode))
            for (const a of node.arguments) actionRanges.push(a.range)
        },
        MemberExpression(node) {
          const name = storeApiName(node)
          if (WRITES.includes(name)) writes.push({ node, name })
        },
        'Program:exit'() {
          for (const { node, name } of writes)
            if (!actionRanges.some((r) => within(node.range, r)))
              stores.push(storeSite(node, name, context))
        },
      }
    },
  }
}

/** The store sites `exceptions` does not cover go into `found`; returns the
 * keys that matched */
function applyExceptions(sites, exceptions, found) {
  const used = new Set()
  for (const { key, site } of sites)
    if (Object.hasOwn(exceptions, key)) used.add(key)
    else found.push(site)
  return [...used].sort()
}

const isWrite = (node) =>
  node.type === 'CallExpression' &&
  isMember(node.callee, 'api', null) &&
  WRITES.includes(node.callee.property.name)

/** `api.<write>` read as a value, not called: handed to something that may
 * call it (`picks.download(api.archiveAssets)`) */
const isWriteValue = (node, parent) =>
  isMember(node, 'api', null) &&
  !node.computed &&
  WRITES.includes(node.property.name) &&
  !(parent?.type === 'CallExpression' && parent.callee === node)

const within = (range, [s, e]) => range[0] >= s && range[1] <= e

/** What a top-level statement declares: a named function or class, or each
 * plain identifier a variable declaration binds. It handles the same
 * statement kinds as src/lib/freshness.test.ts's `declarations` */
function declared(decl) {
  if (
    (decl?.type === 'FunctionDeclaration' ||
      decl?.type === 'ClassDeclaration') &&
    decl.id
  )
    return [{ id: decl.id, node: decl }]
  if (decl?.type !== 'VariableDeclaration') return []
  return decl.declarations
    .filter((d) => d.id.type === 'Identifier')
    .map((d) => ({ id: d.id, node: d }))
}

/** A module's top-level declarations (local name to node) and its exports
 * (exported name to local name) */
function topLevel(program) {
  const tops = new Map()
  const exported = new Map()
  for (const statement of program.body) {
    const isExport = statement.type === 'ExportNamedDeclaration'
    for (const { id, node } of declared(
      isExport ? statement.declaration : statement,
    )) {
      tops.set(id.name, node)
      if (isExport) exported.set(id.name, id.name)
    }
    if (isExport && !statement.declaration && !statement.source)
      for (const sp of statement.specifiers)
        exported.set(sp.exported.name, sp.local.name)
  }
  return { tops, exported }
}

/** Grow `hit` (top-level names) by every top-level declaration whose code
 * references one of them, until nothing changes */
function closeOver(hit, tops, scopeManager) {
  const usedBy = new Map()
  const moduleScope = scopeManager.scopes.find((sc) => sc.type === 'module')
  for (const variable of moduleScope?.variables ?? [])
    usedBy.set(
      variable.name,
      [...tops]
        .filter(([, n]) =>
          variable.references.some((r) => within(r.identifier.range, n.range)),
        )
        .map(([name]) => name),
    )
  const queue = [...hit]
  while (queue.length)
    for (const user of usedBy.get(queue.pop()) ?? [])
      if (!hit.has(user)) {
        hit.add(user)
        queue.push(user)
      }
  return hit
}

/** The names a plain .ts module exports whose code calls an `api` write:
 * directly, or through another of the module's own top-level declarations
 * that does (src/lib/freshness.test.ts's `readingExports`, for writes). A
 * write made through another module is a second level, out of reach */
function writingExports(program, scopeManager, writeRanges) {
  const { tops, exported } = topLevel(program)
  const direct = [...tops]
    .filter(([, n]) => writeRanges.some((r) => within(r, n.range)))
    .map(([name]) => name)
  const writes = closeOver(new Set(direct), tops, scopeManager)
  return new Set(
    [...exported].filter(([, l]) => writes.has(l)).map(([name]) => name),
  )
}

/** The function declarations that are guarded: used, and every use outside
 * their own body inside an `action(` call's arguments. A use is a scope
 * reference, so a property key or a shadowing local is none */
function guardedFunctions(functions, sourceCode, inAction) {
  return new Set(
    functions.filter((fn) => {
      const uses = sourceCode
        .getDeclaredVariables(fn)
        .flatMap((v) => v.references)
        .filter((r) => !within(r.identifier.range, fn.range))
      return uses.length > 0 && uses.every((r) => inAction(r.identifier.range))
    }),
  )
}

const importedName = (spec) =>
  spec.type === 'ImportSpecifier'
    ? (spec.imported.name ?? spec.imported.value)
    : 'default'

/** The member a reference reads, `ns.name`, if it reads one */
function memberOf(id, sourceCode) {
  const parent = sourceCode.getAncestors(id).at(-1)
  return parent?.type === 'MemberExpression' &&
    parent.object === id &&
    !parent.computed
    ? parent
    : null
}

/** Each unguarded use of an import: the node and the exported name it
 * stands for. A namespace import is used through `ns.name` */
function importUses(spec, sourceCode, guarded) {
  if (spec.importKind === 'type') return []
  const namespace = spec.type === 'ImportNamespaceSpecifier'
  const uses = sourceCode
    .getDeclaredVariables(spec)
    .flatMap((v) => v.references)
    .map(({ identifier: id }) => {
      if (!namespace) return { site: id, name: importedName(spec) }
      const member = memberOf(id, sourceCode)
      return member && { site: member, name: member.property.name }
    })
  return uses.filter((u) => u && !guarded(u.site))
}

/** The local functions in `scope` (`const put = (n) => set(n)`) used, and
 * used only inside an `action(` call's arguments */
function guardedLocals(scope, inAction) {
  return scope.variables
    .filter((v) => {
      const init = v.defs.length === 1 ? v.defs[0].node.init : null
      if (!/^(Arrow)?Function/.test(init?.type ?? '')) return false
      const uses = v.references.filter((r) => !r.init)
      return uses.length > 0 && uses.every((r) => inAction(r.identifier.range))
    })
    .map((v) => v.defs[0].node.init.range)
}

/** The parameter indexes of `fn` it hands only to `action(`: every use is
 * inside an `action(` call's arguments, or inside a local function that is */
function guardedParams(fn, sourceCode, inAction) {
  const scope = sourceCode.scopeManager.acquire(fn)
  const held = new Set()
  if (!scope) return held
  const locals = guardedLocals(scope, inAction)
  const safe = (range) =>
    inAction(range) || locals.some((r) => within(range, r))
  fn.params.forEach((param, i) => {
    const refs =
      param.type === 'Identifier'
        ? (scope.set.get(param.name)?.references ?? [])
        : []
    if (refs.length && refs.every((r) => safe(r.identifier.range))) held.add(i)
  })
  return held
}

/** The functions an export declaration holds, keyed as `guards` keys them:
 * a function by its name, a class's method as `Class.method` */
function exportedFunctions(statement) {
  const decl =
    statement.type === 'ExportNamedDeclaration' ? statement.declaration : null
  if (!decl?.id) return []
  if (decl.type === 'FunctionDeclaration') return [[decl.id.name, decl]]
  if (decl.type !== 'ClassDeclaration') return []
  return decl.body.body
    .filter((m) => m.type === 'MethodDefinition' && m.key.type === 'Identifier')
    .map((m) => [`${decl.id.name}.${m.key.name}`, m.value])
}

/** The functions a module exports that guard an argument, each to the
 * parameter indexes it hands only to `action(` (Picks.download) */
function guardingExports(program, sourceCode, inAction) {
  const guards = new Map()
  for (const [key, fn] of program.body.flatMap(exportedFunctions)) {
    const held = guardedParams(fn, sourceCode, inAction)
    if (held.size) guards.set(key, held)
  }
  return guards
}

const API_MODULE = 'src/lib/api.ts'

/** In a .ts module (a `.svelte.ts` store included), record the arguments its
 * exports guard in `guards`; in a plain .ts module, also its writing exports
 * in `writers` - for api.ts, its EXPORTED_WRITES */
function moduleRule(file, sourceCode, { writers, guards }) {
  const writeRanges = []
  const actionRanges = []
  return {
    CallExpression(node) {
      if (isWrite(node)) writeRanges.push(node.range)
      if (isActionCall(node, sourceCode))
        for (const a of node.arguments) actionRanges.push(a.range)
    },
    'Program:exit'(program) {
      const inAction = (range) => actionRanges.some((r) => within(range, r))
      guards.set(file, guardingExports(program, sourceCode, inAction))
      if (file.endsWith('.svelte.ts')) return
      const writing = writingExports(
        program,
        sourceCode.scopeManager,
        writeRanges,
      )
      if (relative(UI, file).replaceAll('\\', '/') === API_MODULE)
        for (const name of EXPORTED_WRITES) writing.add(name)
      writers.set(file, writing)
    },
  }
}

/** Where an identifier was imported from: `[specifier, imported name]` */
function importOf(id, sourceCode) {
  const def = id.type === 'Identifier' && variableOf(sourceCode, id)?.defs[0]
  return def?.type === 'ImportBinding'
    ? [def.parent.source.value, importedName(def.node)]
    : null
}

/** What a call's callee is, as `guards` keys it: an imported function
 * (`[specifier, name]`), or a method of an instance made from an imported
 * class (`picks.download`, `const picks = new Picks()`) */
function calleeExport(callee, sourceCode) {
  if (callee.type === 'Identifier') return importOf(callee, sourceCode)
  const cls = isMember(callee, callee.object?.name, null)
    ? classOf(callee.object, sourceCode)
    : null
  return cls && [cls[0], `${cls[1]}.${callee.property.name}`]
}

/** The imported class a variable was made from (`const picks = new
 * Picks()`), as `[specifier, class name]` */
function classOf(id, sourceCode) {
  const def = variableOf(sourceCode, id)?.defs[0]
  const made = def?.type === 'Variable' ? def.node.init : null
  return made?.type === 'NewExpression'
    ? importOf(made.callee, sourceCode)
    : null
}

/** An `api.<write>` value an `action(` does not guard: a site in `found`,
 * or, when it is an argument of an imported function or method, a site
 * measure resolves against the `guards` that module's exports declare */
function writeValueSite(node, sourceCode, { at, found, guardSites, file }) {
  const site = `${at(node)} ${node.property.name}`
  const call = node.parent
  const index =
    call?.type === 'CallExpression' ? call.arguments.indexOf(node) : -1
  const target = index === -1 ? null : calleeExport(call.callee, sourceCode)
  if (!target) return found.push(site)
  guardSites.push({
    from: file,
    specifier: target[0],
    key: target[1],
    index,
    site,
  })
}

/** `api.<write>` calls and values in a .svelte file that no `action(`
 * guards go into `found` (or `guardSites`, an argument some import may
 * guard), and unguarded uses of imports into `helperSites`; measure
 * resolves both after the lint, against the `writers` and `guards` of every
 * .ts module */
function unguardedRule(found, sites) {
  return {
    create(context) {
      const file = context.filename
      const sourceCode = context.sourceCode
      if (file.endsWith('.ts')) return moduleRule(file, sourceCode, sites)
      if (!file.endsWith('.svelte')) return {}
      const writes = []
      const values = []
      const actionRanges = []
      const functions = []
      const imports = []
      const at = (node) => `${relative(UI, file)}:${node.loc.start.line}`
      return {
        CallExpression(node) {
          if (isActionCall(node, sourceCode))
            for (const a of node.arguments) actionRanges.push(a.range)
          if (isWrite(node)) writes.push(node)
        },
        MemberExpression(node) {
          if (isWriteValue(node, node.parent)) values.push(node)
        },
        FunctionDeclaration(node) {
          if (node.id) functions.push(node)
        },
        ImportDeclaration(node) {
          if (node.importKind !== 'type') imports.push(node)
        },
        'Program:exit'() {
          const inAction = (range) => actionRanges.some((r) => within(range, r))
          const safe = guardedFunctions(functions, sourceCode, inAction)
          const guarded = (node) =>
            inAction(node.range) ||
            sourceCode.getAncestors(node).some((a) => safe.has(a))
          for (const node of writes.filter((n) => !guarded(n)))
            found.push(`${at(node)} ${node.callee.property.name}`)
          const into = { at, found, guardSites: sites.guardSites, file }
          for (const node of values.filter((n) => !guarded(n)))
            writeValueSite(node, sourceCode, into)
          for (const declaration of imports)
            for (const spec of declaration.specifiers)
              for (const { site, name } of importUses(
                spec,
                sourceCode,
                guarded,
              ))
                sites.helperSites.push({
                  from: file,
                  specifier: declaration.source.value,
                  name,
                  site: `${at(site)} ${name}`,
                })
        },
      }
    },
  }
}

/** A module specifier (`import ... from './x'`), which is never copy */
const isModuleSource = (node) =>
  /^(Import|Export(All|Named)?)(Declaration|Expression)$/.test(
    node.parent?.type ?? '',
  ) && node.parent.source === node

/** Copy that matches a banned term, outside COPY_OWNERS */
function copyRule(found) {
  return {
    create(context) {
      if (COPY_OWNERS.has(relative(UI, context.filename).replaceAll('\\', '/')))
        return {}
      const check = (node, text) => {
        for (const { id, pattern } of BANNED)
          if (pattern.test(text))
            found.push(
              `${relative(UI, context.filename)}:${node.loc.start.line} ${id}`,
            )
      }
      return {
        Literal(node) {
          if (typeof node.value === 'string' && !isModuleSource(node))
            check(node, node.value)
        },
        TemplateLiteral(node) {
          const text = node.quasis
            .map((q) => q.value.cooked ?? q.value.raw)
            .join('0')
          check(node, text)
        },
        SvelteText(node) {
          const inStyle = context.sourceCode
            .getAncestors(node)
            .some((a) => a.type === 'SvelteStyleElement')
          if (!inStyle) check(node, node.value)
        },
        SvelteLiteral: (node) => check(node, node.value),
      }
    },
  }
}

export const TRIVIAL_LINE =
  /^([{}()[\];,]*|<\/?[a-z]+>|\/\/.*|\/\*.*|\*.*|\*\/|-->|<!--.*)$/i
export const BLOCK_LINES = 6

/** Per .svelte file sharing any, the lines that lie in a 6-line window
 * another file shares: `[file: N lines, ...]` and their total. Lines, not
 * runs, so a fix in the middle of a run never raises the count */
export function duplicateBlocks(texts) {
  const windows = new Map()
  const owners = new Map()
  const numbers = new Map()
  for (const [file, text] of texts) {
    let lines = text
      .split('\n')
      .map((l, n) => [l.trim().replace(/\s+/g, ' '), n + 1])
      .filter(([l]) => l && !TRIVIAL_LINE.test(l))
    numbers.set(
      file,
      lines.map(([, n]) => n),
    )
    lines = lines.map(([l]) => l)
    const hashes = []
    for (let i = 0; i + BLOCK_LINES <= lines.length; i++) {
      const hash = createHash('sha1')
        .update(lines.slice(i, i + BLOCK_LINES).join('\n'))
        .digest('hex')
      hashes.push(hash)
      if (!owners.has(hash)) owners.set(hash, new Set())
      owners.get(hash).add(file)
    }
    windows.set(file, hashes)
  }
  const files = []
  let lines = 0
  for (const [file, hashes] of windows) {
    const shared = new Set()
    hashes.forEach((hash, i) => {
      if (owners.get(hash).size > 1)
        for (let k = 0; k < BLOCK_LINES; k++) shared.add(i + k)
    })
    if (!shared.size) continue
    const ranges = lineRanges(
      [...shared].sort((a, b) => a - b),
      numbers.get(file),
    )
    files.push(`${relative(UI, file)}: ${shared.size} lines (${ranges})`)
    lines += shared.size
  }
  return { files, lines }
}

/** Kept-line indexes (ascending) as source line ranges, `3-9, 14-20`: a
 * run of consecutive kept lines is one range, whatever blank or trivial
 * lines lie between them in the source */
function lineRanges(indexes, numbers) {
  const ranges = []
  let start = indexes[0]
  indexes.forEach((index, i) => {
    if (indexes[i + 1] === index + 1) return
    const [from, to] = [numbers[start], numbers[index]]
    ranges.push(from === to ? `${from}` : `${from}-${to}`)
    start = indexes[i + 1]
  })
  return ranges.join(', ')
}

/** The unguarded uses the lint left to resolve across modules: a use of a
 * module's writing export, and a write handed to an import that does not
 * guard that argument */
function resolveSites(sites, known) {
  const target = (site) => resolveImport(site.from, site.specifier, known)
  const writes = sites.helperSites.filter((h) =>
    sites.writers.get(target(h))?.has(h.name),
  )
  const unguarded = sites.guardSites.filter(
    (g) => !sites.guards.get(target(g))?.get(g.key)?.has(g.index),
  )
  return [...writes, ...unguarded].map((s) => s.site)
}

/** What is read off each file's text: files over SIZE_CEILING (as
 * `file (N lines)` detail) and over HARD_SIZE_CEILING, a11y suppressions,
 * the .svelte texts duplicate_blocks compares, and CSS backgrounds into
 * `rawMedia` */
function textFacts(files, rawMedia) {
  const oversized = []
  let overHard = 0
  let suppressions = 0
  const svelteTexts = []
  for (const file of files) {
    const text = readFileSync(file, 'utf8')
    if (file.endsWith('.svelte')) svelteTexts.push([file, text])
    rawMedia.push(...cssMedia(file, text))
    const lines = text.split('\n').length - (text.endsWith('\n') ? 1 : 0)
    if (lines > SIZE_CEILING)
      oversized.push(`${relative(UI, file)} (${lines} lines)`)
    if (lines > HARD_SIZE_CEILING) overHard++
    suppressions += (text.match(/svelte-ignore[^\n]*\ba11y_/g) ?? []).length
  }
  return {
    oversized: oversized.sort(byPlace),
    overHard,
    suppressions,
    svelteTexts,
  }
}

/** One ESLint finding as `file:line name (value)`: the name is what the
 * rule's message calls the function (`Function 'f'`, `Arrow function`),
 * the value the first number it reports (complexity, or line count) */
function functionSite(result, message) {
  const name = message.message.split(' has ')[0]
  const value = message.message.match(/\d+/)?.[0]
  return `${relative(UI, result.filePath)}:${message.line} ${name} (${value})`
}

export async function measure(
  files = sourceFiles(),
  prefixes = referencePrefixes(),
  lib = LIB,
) {
  const known = new Set(files)
  const imports = {}
  const edges = []
  const prefixHits = []
  const rawToasts = []
  const emptyImports = []
  const unwatched = []
  const storeReads = []
  const storeWrites = []
  const dialogs = []
  const unguarded = []
  const sites = {
    helperSites: [],
    guardSites: [],
    writers: new Map(),
    guards: new Map(),
  }
  const banned = []
  const rawMedia = []
  const collector = {
    rules: {
      dialogs: {
        create(context) {
          return {
            CallExpression(node) {
              if (isWindowDialog(node.callee))
                dialogs.push(
                  `${relative(UI, context.filename)}:${node.loc.start.line}`,
                )
            },
          }
        },
      },
      mutations: unguardedRule(unguarded, sites),
      copy: copyRule(banned),
      media: mediaRule(rawMedia),
      toasts: {
        create(context) {
          return {
            CallExpression(node) {
              if (!isMember(node.callee, 'notify', 'error')) return
              const first = node.arguments[0]
              if (first && isRawError(context.sourceCode, node, first))
                rawToasts.push(
                  `${relative(UI, context.filename)}:${node.loc.start.line}`,
                )
            },
          }
        },
      },
      empty: {
        create(context) {
          if (isLoadState(context.filename)) return {}
          return {
            ImportDeclaration(node) {
              if (String(node.source.value).endsWith('/Empty.svelte'))
                emptyImports.push(relative(UI, context.filename))
            },
          }
        },
      },
      reads: readsRule(unwatched, storeReads),
      storeWrites: storeWritesRule(storeWrites),
      imports: {
        create(context) {
          const from = context.filename
          imports[from] ??= []
          return {
            ImportDeclaration(node) {
              const target = resolveImport(from, node.source.value, known)
              if (!target) return
              imports[from].push(target)
              edges.push({ from, line: node.loc.start.line, target })
            },
          }
        },
      },
      prefixes: {
        create(context) {
          const owner = PREFIX_OWNERS.has(relative(UI, context.filename))
          const check = (node, value) => {
            if (
              !owner &&
              typeof value === 'string' &&
              prefixes.some((p) => value.startsWith(p))
            )
              prefixHits.push(
                `${relative(UI, context.filename)}:${node.loc.start.line}`,
              )
          }
          return {
            Literal: (node) => check(node, node.value),
            TemplateElement: (node) => check(node, node.value.cooked),
            SvelteLiteral: (node) => check(node, node.value),
          }
        },
      },
    },
  }
  const eslint = new ESLint({
    cwd: UI,
    // A branch's own comments and ignore patterns must not hide what is
    // measured: an inline `eslint-disable complexity` or a new `ignores`
    // entry would otherwise drop a function or a file from the count
    allowInlineConfig: false,
    ignore: false,
    overrideConfig: [
      {
        plugins: { arch: collector },
        rules: {
          complexity: ['error', COMPLEXITY_LIMIT],
          'max-lines-per-function': [
            'error',
            {
              max: FUNCTION_LINE_LIMIT,
              skipBlankLines: true,
              skipComments: true,
            },
          ],
          'arch/imports': 'error',
          'arch/prefixes': 'error',
          'arch/toasts': 'error',
          'arch/empty': 'error',
          'arch/reads': 'error',
          'arch/dialogs': 'error',
          'arch/mutations': 'error',
          'arch/storeWrites': 'error',
          'arch/copy': 'error',
          'arch/media': 'error',
        },
      },
    ],
  })
  const scripts = files.filter((f) => !f.endsWith('.css'))
  const results = await eslint.lintFiles(scripts)
  // A file that does not parse reports no rule at all; counted as zero it
  // would read as an improvement
  const unparsed = results.filter((r) => r.messages.some((m) => m.fatal))
  if (unparsed.length)
    throw new Error(
      `could not parse: ${unparsed.map((r) => relative(UI, r.filePath)).join(', ')}`,
    )
  const count = (rule) =>
    results.reduce(
      (n, r) => n + r.messages.filter((m) => m.ruleId === rule).length,
      0,
    )
  const sitesOf = (rule) =>
    results
      .flatMap((r) =>
        r.messages
          .filter((m) => m.ruleId === rule)
          .map((m) => functionSite(r, m)),
      )
      .sort(byPlace)

  unguarded.push(...resolveSites(sites, known))
  const excepted = {
    reads: applyExceptions(storeReads, STORE_EXCEPTIONS.reads, unwatched),
    writes: applyExceptions(storeWrites, STORE_EXCEPTIONS.writes, unguarded),
  }
  unguarded.sort(byPlace)

  const graph = {}
  for (const file of scripts) graph[file] = [...new Set(imports[file] ?? [])]
  const components = cycles(graph)
  const layering = layeringViolations(edges, lib)

  const { oversized, overHard, suppressions, svelteTexts } = textFacts(
    files,
    rawMedia,
  )
  const duplicates = duplicateBlocks(svelteTexts)
  banned.sort(byPlace)
  rawMedia.sort(byPlace)
  return {
    metrics: {
      files_over_size_ceiling: oversized.length,
      files_over_hard_ceiling: overHard,
      complex_functions: count('complexity'),
      long_functions: count('max-lines-per-function'),
      import_cycles: components.length,
      modules_in_import_cycles: components.reduce((n, c) => n + c.length, 0),
      layering_violations: layering.length,
      prefix_literals: prefixHits.length,
      a11y_suppressions: suppressions,
      raw_error_toasts: rawToasts.length,
      empty_outside_loadstate: emptyImports.length,
      unwatched_reads: unwatched.length,
      window_dialogs: dialogs.length,
      unguarded_mutations: unguarded.length,
      banned_terms: banned.length,
      raw_media: rawMedia.length,
      duplicate_blocks: duplicates.lines,
    },
    detail: {
      files_over_size_ceiling: oversized,
      complex_functions: sitesOf('complexity'),
      long_functions: sitesOf('max-lines-per-function'),
      cycles: components.map((c) => c.map((f) => relative(UI, f))),
      layering_violations: layering,
      prefix_literals: prefixHits,
      raw_error_toasts: rawToasts,
      empty_outside_loadstate: emptyImports,
      unwatched_reads: unwatched,
      store_exceptions: excepted,
      window_dialogs: dialogs,
      unguarded_mutations: unguarded,
      banned_terms: banned,
      raw_media: rawMedia,
      duplicate_blocks: duplicates.files,
    },
  }
}

/** Detail order: by file, then line (numerically), then what was found */
const byPlace = (a, b) => a.localeCompare(b, 'en', { numeric: true })

/** The proxies: counts of shape, not of a broken rule. They are measured
 * and recorded like the rest, but a rise only warns (on stderr), never
 * fails --check or --compare */
export const ADVISORY = new Set([
  'files_over_size_ceiling',
  'complex_functions',
  'long_functions',
  'duplicate_blocks',
])

const rises = (current, baseline) =>
  Object.entries(current).filter(
    ([key, value]) => key in baseline && value > baseline[key],
  )

/** The enforced metrics that rose, as `key: base -> cur` - the line
 * harnest's check_ui and its waiver parse */
export function regressions(current, baseline) {
  return rises(current, baseline)
    .filter(([key]) => !ADVISORY.has(key))
    .map(([key, value]) => `${key}: ${baseline[key]} -> ${value}`)
}

/** The advisory metrics that rose: `{ key, line }`, line as regressions
 * words it */
export function advisories(current, baseline) {
  return rises(current, baseline)
    .filter(([key]) => ADVISORY.has(key))
    .map(([key, value]) => ({
      key,
      line: `${key}: ${baseline[key]} -> ${value}`,
    }))
}

/** Each advisory rise to stderr, `warning (advisory): ...`, followed by
 * the items detail names for it (under --compare, the current
 * measurement's own `detail`, when it carries one) */
function warnAdvisories(current, baseline, detail = {}) {
  for (const { key, line } of advisories(current, baseline)) {
    console.error(`warning (advisory): ${line}`)
    for (const item of detail[key] ?? []) console.error(`  ${item}`)
  }
}

async function main(argv) {
  // Two measurements already taken (harnest measures a merge base and a
  // branch, then asks this script's rule which metrics rose)
  // Exit 1 means "rose" and nothing else: input it cannot read is 2, so a
  // caller never mistakes a failure for a regression or a pass
  if (argv[0] === '--compare') {
    let current, baseline
    try {
      ;[current, baseline] = argv
        .slice(1, 3)
        .map((path) => JSON.parse(readFileSync(path, 'utf8')))
    } catch (e) {
      console.error(`--compare: ${e instanceof Error ? e.message : e}`)
      return 2
    }
    const problems = regressions(current, baseline)
    for (const line of problems) console.log(line)
    warnAdvisories(current, baseline, current.detail)
    return problems.length ? 1 : 0
  }
  const { metrics, detail } = await measure()
  const flag = argv[0]
  if (flag === '--write') {
    writeFileSync(argv[1], JSON.stringify(metrics, null, 2) + '\n')
  } else if (flag === '--check') {
    const baseline = JSON.parse(readFileSync(argv[1], 'utf8'))
    const problems = regressions(metrics, baseline)
    for (const line of problems) console.log(line)
    warnAdvisories(metrics, baseline, detail)
    if (problems.length) return 1
  }
  console.log(JSON.stringify({ ...metrics, detail }, null, 2))
  return 0
}

if (process.argv[1] === fileURLToPath(import.meta.url))
  process.exit(await main(process.argv.slice(2)))
