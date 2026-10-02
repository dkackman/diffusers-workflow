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
import { ESLint } from 'eslint'
import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs'
import { dirname, join, relative, resolve, extname } from 'node:path'
import { fileURLToPath } from 'node:url'

const UI = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const SRC = join(UI, 'src')
const REFERENCES_PY = join(UI, '..', 'dw', 'references.py')

export const SIZE_CEILING = 600
export const COMPLEXITY_LIMIT = 10
export const FUNCTION_LINE_LIMIT = 60
export const PREFIX_OWNERS = new Set(['src/lib/references.ts'])

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

export async function measure(
  files = sourceFiles(),
  prefixes = referencePrefixes(),
) {
  const known = new Set(files)
  const imports = {}
  const prefixHits = []
  const collector = {
    rules: {
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
  for (const file of files) {
    const text = readFileSync(file, 'utf8')
    if (text.split('\n').length - (text.endsWith('\n') ? 1 : 0) > SIZE_CEILING)
      oversized++
    suppressions += (text.match(/svelte-ignore[^\n]*\ba11y_/g) ?? []).length
  }
  return {
    metrics: {
      files_over_size_ceiling: oversized,
      complex_functions: count('complexity'),
      long_functions: count('max-lines-per-function'),
      import_cycles: components.length,
      modules_in_import_cycles: components.reduce((n, c) => n + c.length, 0),
      prefix_literals: prefixHits.length,
      a11y_suppressions: suppressions,
    },
    detail: {
      cycles: components.map((c) => c.map((f) => relative(UI, f))),
      prefix_literals: prefixHits,
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
