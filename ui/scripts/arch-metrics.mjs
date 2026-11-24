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
// Counting rules, fixed so any commit measures the same way:
// - Sources are ui/src's .ts, .svelte and .css files, tests excluded
//   (*.test.ts, *.svelte.test.ts).
// - files_over_size_ceiling: files above SIZE_CEILING raw lines; a .svelte
//   file's <style> counts, since it is part of what a reader holds.
// - complex_functions: ESLint's core `complexity` rule above
//   COMPLEXITY_LIMIT, script blocks only - ESLint does not see template
//   {#if}/{#each} branching.
// - long_functions: ESLint's `max-lines-per-function` above
//   FUNCTION_LINE_LIMIT, blank and comment lines skipped.
// - import_cycles / modules_in_import_cycles: strongly connected components
//   of more than one module over relative imports, type-only imports
//   included (they still couple the files).
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
// - unwatched_reads: in a .svelte file, `api.<read>(...)` calls with no
//   `resource(` or `poll(` call around them. A read is `get*`, `list*` or a
//   name in READ_ALIASES. Every other `api` key is in NOT_READS, and
//   arch-metrics.test.ts fails, naming the key, when one is in neither list
//   or in both - so a new method cannot dodge this count unclassified.
// Known limits of those three rules:
// - `resource` and `poll` are matched by callee name.
// - `api` is matched by identifier name, so an alias or destructuring is
//   invisible.
// - `.svelte.ts` stores are not measured for reads.
// - a read made through a plain `.ts` helper is not counted here.
//   src/lib/freshness.test.ts sees one import level: a .svelte file that
//   imports a src/lib .ts module calling an `api` read must have a
//   `resource(` or be a declared exception. A read two or more imports
//   away is out of reach of both checks.
// - an `Empty` re-exported through a barrel is not counted.
// - `String(...)` matches a single identifier only.
// - window_dialogs: calls to `window.confirm`, `window.prompt`,
//   `window.alert`, or a bare `alert`, `confirm` or `prompt` - a dialog is
//   `confirmDialog`/`chooseDialog` (src/lib/confirm.svelte.ts).
// - unguarded_mutations: in a .svelte file, an `api.<write>` call (WRITES),
//   or a use of an imported plain .ts function whose code calls one, lexically
//   inside no `action(` call's arguments and inside no guarded function
//   declaration. A function declaration is guarded when it is used and every
//   use of it outside its own body (a scope reference, so neither a property
//   key nor a shadowing local) is inside an `action(` call's arguments: a
//   helper also bound to `onclick={helper}` is not. A .ts function writes
//   when its code calls an `api` write directly or through another of its
//   module's own top-level declarations - one import level, as
//   src/lib/freshness.test.ts follows reads.
// - banned_terms: user-facing copy matching a BANNED pattern
//   (src/lib/terms.ts), outside COPY_OWNERS. Copy is a string literal with
//   whitespace in it, a template literal whose parts joined with "0" have
//   whitespace, Svelte markup text outside <style>, and a static attribute
//   value. Each pattern counts once per piece of copy.
// - duplicate_blocks: over .svelte files, lines trimmed and their
//   whitespace collapsed, blank and TRIVIAL_LINE lines dropped; a 6-line
//   window whose sha1 occurs in two or more files is shared, and each
//   file's maximal runs of consecutive shared windows are counted.
// - raw_media: media elements outside RAW_MEDIA_OWNERS (OutputTile and
//   MediaBody), in .svelte and .ts files; a media file is shown through
//   them. Counted forms: `<img>`, `<video>`, `<audio>`, `<source>`,
//   `<picture>`; `<svelte:element this="img">` with a static `this` (a
//   dynamic `this={expr}` is not resolved); `X.createElement('img')` (also
//   video, audio, source, picture) with a static name; `new Audio()` and
//   `new Image()`; and `{@html expr}` whose expression text holds one of
//   those tags. Detail: `file:line form`.
// Known limits of those four rules:
// - a write two or more imports away (a .ts helper that writes through
//   another module) is not counted, nor a write through a `.svelte.ts`
//   store (enhancer, planStore, runQueue): their exports are classes and
//   factories whose methods write, so a use is not a write.
// - a write api.ts exports outside the `api` object (`setNote`) is not
//   counted.
// - only `function` declarations can be guarded helpers; a write inside an
//   arrow helper (`const f = () => api.x()`) counts even when an action
//   names it - name it in the action, don't rename it to `function`.
// - copy assembled from single words, or across `{...}` in markup, can slip
//   through banned_terms.
import { ESLint } from 'eslint'
import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs'
import { dirname, join, relative, resolve, extname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { BANNED, COPY_OWNERS } from '../src/lib/terms.ts'

const UI = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const SRC = join(UI, 'src')
const REFERENCES_PY = join(UI, '..', 'dw', 'references.py')

export const SIZE_CEILING = 600
export const COMPLEXITY_LIMIT = 10
export const FUNCTION_LINE_LIMIT = 60
export const PREFIX_OWNERS = new Set(['src/lib/references.ts'])

export const LOAD_STATE = 'src/lib/ui/LoadState.svelte'
export const READ_ALIASES = [
  'gallery',
  'galleryMetadata',
  'health',
  'memory',
  'diffusersStatus',
  'server',
  'eventLog',
  'describeTask',
  'describePipeline',
  'describeClass',
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
const MEDIA_TAGS = new Set(['img', 'video', 'audio', 'source', 'picture'])
const MEDIA_HTML = /<\s*(img|video|audio|source|picture)\b/i
const MEDIA_CONSTRUCTORS = new Set(['Audio', 'Image'])

/** The static text of a string literal or an expression-free template */
function staticText(node) {
  if (node?.type === 'Literal' && typeof node.value === 'string')
    return node.value
  if (node?.type === 'TemplateLiteral' && node.expressions.length === 0)
    return node.quasis.map((q) => q.value.cooked ?? q.value.raw).join('')
  return null
}

/** Media elements outside RAW_MEDIA_OWNERS, however they are written: markup,
 * `<svelte:element this="img">`, `document.createElement('video')`,
 * `new Audio()` / `new Image()`, and an `{@html}` string holding a media tag */
function mediaRule(found) {
  return {
    create(context) {
      if (
        RAW_MEDIA_OWNERS.has(
          relative(UI, context.filename).replaceAll('\\', '/'),
        )
      )
        return {}
      const add = (node, what) =>
        found.push(
          `${relative(UI, context.filename)}:${node.loc.start.line} ${what}`,
        )
      return {
        SvelteElement(node) {
          const tag = node.name?.name
          if (MEDIA_TAGS.has(tag)) return add(node, tag)
          if (tag !== 'svelte:element') return
          const attr = node.startTag?.attributes?.find(
            (a) => a.type === 'SvelteAttribute' && a.key?.name === 'this',
          )
          const v = attr?.value?.[0]
          const name = v?.type === 'SvelteLiteral' ? v.value : null
          if (MEDIA_TAGS.has(name)) add(node, name)
        },
        SvelteMustacheTag(node) {
          if (
            node.kind === 'raw' &&
            MEDIA_HTML.test(context.sourceCode.getText(node.expression))
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

export const isLoadState = (file) =>
  relative(UI, file).replaceAll('\\', '/') === LOAD_STATE

const RAW_ERROR = /\.message\b|String\(\s*\w+\s*\)/
// The text a toast argument stands for: an identifier is followed to its
// `const` (or single-assignment `let`) initializer, through enclosing scopes.
function isRawError(sourceCode, call, arg) {
  if (RAW_ERROR.test(sourceCode.getText(arg))) return true
  if (arg.type !== 'Identifier') return false
  for (let s = sourceCode.getScope(call); s; s = s.upper) {
    const v = s.set.get(arg.name)
    if (!v) continue
    const def = v.defs.length === 1 ? v.defs[0] : null
    const decl = def?.node
    if (def?.type !== 'Variable' || !decl?.init) return false
    const writes = v.references.filter((r) => r.isWrite()).length
    if (def.parent?.kind !== 'const' && writes !== 1) return false
    return RAW_ERROR.test(sourceCode.getText(decl.init))
  }
  return false
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

function resolveImport(from, specifier, known) {
  if (!specifier.startsWith('.')) return null
  const base = resolve(dirname(from), specifier)
  for (const candidate of [base, `${base}.ts`, join(base, 'index.ts')])
    if (known.has(candidate)) return candidate
  return null
}

const DIALOG_NAMES = new Set(['alert', 'confirm', 'prompt'])
const isWindowDialog = (callee) =>
  (callee.type === 'Identifier' && DIALOG_NAMES.has(callee.name)) ||
  (isMember(callee, 'window', null) && DIALOG_NAMES.has(callee.property.name))

const isActionCall = (node) =>
  node.type === 'CallExpression' &&
  node.callee.type === 'Identifier' &&
  node.callee.name === 'action'

const isWrite = (node) =>
  node.type === 'CallExpression' &&
  isMember(node.callee, 'api', null) &&
  WRITES.includes(node.callee.property.name)

const within = (range, [s, e]) => range[0] >= s && range[1] <= e

/** A module's top-level declarations (local name to node) and its exports
 * (exported name to local name) */
function topLevel(program) {
  const tops = new Map()
  const exported = new Map()
  const declare = (id, node, isExport) => {
    tops.set(id.name, node)
    if (isExport) exported.set(id.name, id.name)
  }
  const declareAll = (decl, isExport) =>
    decl.declarations
      .filter((d) => d.id.type === 'Identifier')
      .forEach((d) => declare(d.id, d, isExport))
  for (const statement of program.body) {
    const isExport = statement.type === 'ExportNamedDeclaration'
    const decl = isExport ? statement.declaration : statement
    if (decl?.type === 'FunctionDeclaration' && decl.id)
      declare(decl.id, decl, isExport)
    else if (decl?.type === 'VariableDeclaration') declareAll(decl, isExport)
    else if (isExport && !statement.source)
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

/** In a plain .ts module, record its writing exports in `writers` */
function writersRule(file, sourceCode, writers) {
  const writeRanges = []
  return {
    CallExpression(node) {
      if (isWrite(node)) writeRanges.push(node.range)
    },
    'Program:exit'(program) {
      writers.set(
        file,
        writingExports(program, sourceCode.scopeManager, writeRanges),
      )
    },
  }
}

/** `api.<write>` calls in a .svelte file that no `action(` guards go into
 * `found`, and unguarded uses of imports into `helperSites`, which measure
 * resolves after the lint against `writers` - the writing exports of every
 * plain .ts module */
function unguardedRule(found, helperSites, writers) {
  return {
    create(context) {
      const file = context.filename
      const sourceCode = context.sourceCode
      if (file.endsWith('.ts') && !file.endsWith('.svelte.ts'))
        return writersRule(file, sourceCode, writers)
      if (!file.endsWith('.svelte')) return {}
      const writes = []
      const actionRanges = []
      const functions = []
      const imports = []
      const at = (node) => `${relative(UI, file)}:${node.loc.start.line}`
      return {
        CallExpression(node) {
          if (isActionCall(node))
            for (const a of node.arguments) actionRanges.push(a.range)
          if (isWrite(node)) writes.push(node)
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
          for (const declaration of imports)
            for (const spec of declaration.specifiers)
              for (const { site, name } of importUses(
                spec,
                sourceCode,
                guarded,
              ))
                helperSites.push({
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
          if (typeof node.value === 'string' && /\s/.test(node.value))
            check(node, node.value)
        },
        TemplateLiteral(node) {
          const text = node.quasis
            .map((q) => q.value.cooked ?? q.value.raw)
            .join('0')
          if (/\s/.test(text)) check(node, text)
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

/** Per .svelte file, the maximal runs of 6-line windows another file shares */
export function duplicateBlocks(texts) {
  const windows = new Map()
  const owners = new Map()
  for (const [file, text] of texts) {
    const lines = text
      .split('\n')
      .map((l) => l.trim().replace(/\s+/g, ' '))
      .filter((l) => l && !TRIVIAL_LINE.test(l))
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
  const runs = []
  for (const [file, hashes] of windows) {
    let inRun = false
    hashes.forEach((hash, i) => {
      const shared = owners.get(hash).size > 1
      if (shared && !inRun) runs.push(`${relative(UI, file)}:window ${i + 1}`)
      inRun = shared
    })
  }
  return runs
}

export async function measure(
  files = sourceFiles(),
  prefixes = referencePrefixes(),
) {
  const known = new Set(files)
  const imports = {}
  const prefixHits = []
  const rawToasts = []
  const emptyImports = []
  const unwatched = []
  const dialogs = []
  const unguarded = []
  const helperSites = []
  const writers = new Map()
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
      mutations: unguardedRule(unguarded, helperSites, writers),
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
      reads: {
        create(context) {
          if (!context.filename.endsWith('.svelte')) return {}
          return {
            CallExpression(node) {
              const callee = node.callee
              if (
                !isMember(callee, 'api', null) ||
                !isRead(callee.property.name)
              )
                return
              const watched = context.sourceCode
                .getAncestors(node)
                .some(
                  (a) =>
                    a.type === 'CallExpression' &&
                    a.callee.type === 'Identifier' &&
                    (a.callee.name === 'resource' || a.callee.name === 'poll'),
                )
              if (!watched) unwatched.push(relative(UI, context.filename))
            },
          }
        },
      },
      imports: {
        create(context) {
          const from = context.filename
          imports[from] ??= []
          return {
            ImportDeclaration(node) {
              const target = resolveImport(from, node.source.value, known)
              if (target) imports[from].push(target)
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

  for (const { from, specifier, name, site } of helperSites) {
    const target = resolveImport(from, specifier, known)
    if (target && writers.get(target)?.has(name)) unguarded.push(site)
  }
  unguarded.sort((a, b) => a.localeCompare(b, 'en', { numeric: true }))

  const graph = {}
  for (const file of scripts) graph[file] = [...new Set(imports[file] ?? [])]
  const components = cycles(graph)

  let oversized = 0
  let suppressions = 0
  const svelteTexts = []
  for (const file of files) {
    const text = readFileSync(file, 'utf8')
    if (file.endsWith('.svelte')) svelteTexts.push([file, text])
    if (text.split('\n').length - (text.endsWith('\n') ? 1 : 0) > SIZE_CEILING)
      oversized++
    suppressions += (text.match(/svelte-ignore[^\n]*\ba11y_/g) ?? []).length
  }
  const duplicates = duplicateBlocks(svelteTexts)
  return {
    metrics: {
      files_over_size_ceiling: oversized,
      complex_functions: count('complexity'),
      long_functions: count('max-lines-per-function'),
      import_cycles: components.length,
      modules_in_import_cycles: components.reduce((n, c) => n + c.length, 0),
      prefix_literals: prefixHits.length,
      a11y_suppressions: suppressions,
      raw_error_toasts: rawToasts.length,
      empty_outside_loadstate: emptyImports.length,
      unwatched_reads: unwatched.length,
      window_dialogs: dialogs.length,
      unguarded_mutations: unguarded.length,
      banned_terms: banned.length,
      raw_media: rawMedia.length,
      duplicate_blocks: duplicates.length,
    },
    detail: {
      cycles: components.map((c) => c.map((f) => relative(UI, f))),
      prefix_literals: prefixHits,
      raw_error_toasts: rawToasts,
      empty_outside_loadstate: emptyImports,
      unwatched_reads: unwatched,
      window_dialogs: dialogs,
      unguarded_mutations: unguarded,
      banned_terms: banned,
      raw_media: rawMedia,
      duplicate_blocks: duplicates,
    },
  }
}

export function regressions(current, baseline) {
  return Object.entries(current)
    .filter(([key, value]) => key in baseline && value > baseline[key])
    .map(([key, value]) => `${key}: ${baseline[key]} -> ${value}`)
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
    if (problems.length) return 1
  }
  console.log(JSON.stringify({ ...metrics, detail }, null, 2))
  return 0
}

if (process.argv[1] === fileURLToPath(import.meta.url))
  process.exit(await main(process.argv.slice(2)))
