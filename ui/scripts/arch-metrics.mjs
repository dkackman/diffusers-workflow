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
// - unguarded_mutations: in a .svelte file, an `api.<write>` call (WRITES)
//   lexically inside no `action(` call and inside no function declaration
//   whose name an `action(` call's arguments reference in the same file -
//   one level, so a helper an action calls counts as guarded.
// - banned_terms: user-facing copy matching a BANNED pattern
//   (src/lib/terms.ts), outside COPY_OWNERS. Copy is a string literal with
//   whitespace in it, a template literal whose parts joined with "0" have
//   whitespace, Svelte markup text outside <style>, and a static attribute
//   value. Each pattern counts once per piece of copy.
// - duplicate_blocks: over .svelte files, lines trimmed and their
//   whitespace collapsed, blank and TRIVIAL_LINE lines dropped; a 6-line
//   window whose sha1 occurs in two or more files is shared, and each
//   file's maximal runs of consecutive shared windows are counted.
// Known limits of those four rules:
// - a write made in a .ts helper (workspaceActions.ts, uploads.ts,
//   promptFile.ts) or through a named import (`setNote`) is not counted.
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
  // URL builders: no request is made
  'exportZipUrl',
  'galleryThumbnailUrl',
  'outputDownloadUrl',
  'workflowDownloadUrl',
  'promptDownloadUrl',
]

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

/** `api.<write>` calls in a .svelte file that no `action(` guards */
function unguardedRule(found) {
  return {
    create(context) {
      if (!context.filename.endsWith('.svelte')) return {}
      const writes = []
      const actionRanges = []
      const identifiers = []
      return {
        CallExpression(node) {
          if (isActionCall(node))
            for (const a of node.arguments) actionRanges.push(a.range)
          const callee = node.callee
          if (
            isMember(callee, 'api', null) &&
            WRITES.includes(callee.property.name)
          )
            writes.push({
              node,
              ancestors: context.sourceCode.getAncestors(node),
            })
        },
        Identifier(node) {
          identifiers.push(node)
        },
        'Program:exit'() {
          const inAction = (range) =>
            actionRanges.some(([s, e]) => range[0] >= s && range[1] <= e)
          const named = new Set(
            identifiers.filter((i) => inAction(i.range)).map((i) => i.name),
          )
          for (const { node, ancestors } of writes) {
            const guarded =
              inAction(node.range) ||
              ancestors.some(
                (a) =>
                  a.type === 'FunctionDeclaration' && named.has(a.id?.name),
              )
            if (!guarded)
              found.push(
                `${relative(UI, context.filename)}:${node.loc.start.line} ${node.callee.property.name}`,
              )
          }
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
  const banned = []
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
      mutations: unguardedRule(unguarded),
      copy: copyRule(banned),
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
