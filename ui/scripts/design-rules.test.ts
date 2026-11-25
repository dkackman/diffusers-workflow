// @vitest-environment node
// Where the --live state colour and the --select place colour may appear,
// held mechanically (ui/CLAUDE.md; app.css's header says what each means)
import { readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, it } from 'vitest'
import { sourceFiles } from './arch-metrics.mjs'

const UI = join(dirname(fileURLToPath(import.meta.url)), '..')

// The places --live may appear: the header's running link and VRAM
// pressure, a running row and status chip (frames.css: `.row-live`;
// app.css: `.chip.running`), the bar a changed tile or card shows
// (frames.css), the job page's progress bar and current step, the flow view's active step and
// segment and its legend's key to them, a model download in flight, and a
// request's progress bar
const LIVE_FILES = [
  'src/App.svelte',
  'src/app.css',
  'src/frames.css',
  'src/lib/editor/FlowLegend.svelte',
  'src/lib/editor/FlowSegments.svelte',
  'src/lib/editor/FlowView.svelte',
  'src/lib/job/JobProgress.svelte',
  'src/lib/pages/ModelsPage.svelte',
  'src/lib/request/RequestProgress.svelte',
]

it('uses --live only where machine state is shown', () => {
  const using = sourceFiles()
    .filter((f) => readFileSync(f, 'utf8').includes('var(--live)'))
    .map((f) => f.slice(UI.length + 1).replaceAll('\\', '/'))
  expect(using).toEqual(LIVE_FILES)
})

// The places --select may appear: the focus ring and the sidebar's open
// workspace and active section. It is chrome only - never a surface behind
// an image
const SELECT_FILES = ['src/app.css', 'src/lib/Sidebar.svelte']

it('uses --select only for the focus ring and the nav', () => {
  const using = sourceFiles()
    .filter((f) => readFileSync(f, 'utf8').includes('var(--select)'))
    .map((f) => f.slice(UI.length + 1).replaceAll('\\', '/'))
  expect(using).toEqual(SELECT_FILES)
})

// A replace that skips the router would skip its leave guards and its
// history stamp; everything else calls replaceRoute
it('location.replace appears only in the router', () => {
  const using = sourceFiles()
    .filter((f) => readFileSync(f, 'utf8').includes('location.replace('))
    .map((f) => f.slice(UI.length + 1).replaceAll('\\', '/'))
  expect(using).toEqual(['src/lib/router.svelte.ts'])
})

// What stands in for a picture is one map: `kindGlyph` in src/lib/media.ts.
// A second spelling of a glyph is a second map to keep in step
it('the kind glyphs are spelled only in media.ts', () => {
  const using = sourceFiles()
    .filter((f) => /[♪¶]/.test(readFileSync(f, 'utf8')))
    .map((f) => f.slice(UI.length + 1).replaceAll('\\', '/'))
  expect(using).toEqual(['src/lib/media.ts'])
})

// One look for a chosen segment and one for a chosen chip: the selected
// styles live in Segmented.svelte and patterns.css, so no other component
// restyles `aria-pressed` or `.activebtn`
it('the segmented and chip selected styles live only in Segmented.svelte and patterns.css', () => {
  const declaring = sourceFiles()
    .filter((f) => /aria-pressed|\.activebtn/.test(styleOf(f)))
    .map((f) => f.slice(UI.length + 1).replaceAll('\\', '/'))
  // the rule found its own declarations, so it is not passing on nothing
  expect(declaring).toEqual(
    expect.arrayContaining(['src/lib/ui/Segmented.svelte', 'src/patterns.css']),
  )
  expect(declaring.filter((f) => f !== 'src/lib/ui/Segmented.svelte')).toEqual(
    declaring.filter((f) => f === 'src/patterns.css'),
  )
})

// A job's status colours are one rule: `.chip.<status>` in app.css, worn
// through StatusChip. A component that restyled one would give a status a
// second look (a cancelled take once showed as a grey idle chip)
it('job status colours are declared only in StatusChip.svelte and app.css', () => {
  const STATUS = /\.chip\.(running|queued|succeeded|failed|cancelled)\b/
  const declaring = sourceFiles()
    .filter((f) => STATUS.test(styleOf(f)))
    .map((f) => f.slice(UI.length + 1).replaceAll('\\', '/'))
  // the rule found the declarations it allows, so it is not passing on nothing
  expect(declaring).toContain('src/app.css')
  expect(
    declaring.filter(
      (f) => f !== 'src/app.css' && f !== 'src/lib/ui/StatusChip.svelte',
    ),
  ).toEqual([])
  // and the chip itself carries the status as its class
  expect(
    readFileSync(join(UI, 'src/lib/ui/StatusChip.svelte'), 'utf8'),
  ).toContain('class="chip {status}"')
})

// The CSS a file declares: a .svelte file's <style> blocks, or a .css file whole
function styleOf(file: string): string {
  const text = readFileSync(file, 'utf8')
  if (file.endsWith('.css')) return text
  if (!file.endsWith('.svelte')) return ''
  return [...text.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/g)]
    .map((m) => m[1])
    .join('\n')
}

// A var(--x) that names nothing silently falls back (or to nothing). A
// custom property is defined for every file by a `:root` rule; for one file
// by its own `--x:` or `style:--x`, or by a component that imports it (the
// parent's value is inherited). A knob with a fallback, `var(--x, 1rem)`, is
// defined when any file sets it. A templated name (`var(--tone-${t})`) is
// not a name and is skipped, and Bits UI's `--bits-*` are its own
const TOKEN = /(?<![\w-])(--[a-zA-Z0-9_-]+)\s*:/g
const tokensIn = (text: string) => [...text.matchAll(TOKEN)].map((m) => m[1])

/** The relative modules a file imports, as absolute paths */
const importsOf = (file: string, text: string) =>
  [...text.matchAll(/from\s+['"](\.[^'"]+)['"]/g)].map((m) =>
    resolve(dirname(file), m[1]),
  )

/** Each `file: --x` that names no custom property defined for it */
function undefinedTokens(sources: [string, string][]): string[] {
  const root = new Set<string>()
  const anywhere = new Set<string>()
  const own = new Map<string, Set<string>>()
  for (const [file, text] of sources) {
    for (const block of text.matchAll(/:root\b[^{}]*\{([^{}]*)\}/g))
      for (const name of tokensIn(block[1])) root.add(name)
    const mine = new Set(tokensIn(text))
    for (const m of text.matchAll(/style:(--[a-zA-Z0-9_-]+)/g)) mine.add(m[1])
    own.set(file, mine)
    for (const name of mine) anywhere.add(name)
  }
  const parents = new Map<string, string[]>()
  for (const [file, text] of sources)
    for (const child of importsOf(file, text))
      parents.set(child, [...(parents.get(child) ?? []), file])
  const definedFor = (file: string, name: string, fallback: boolean) =>
    root.has(name) ||
    own.get(file)?.has(name) ||
    (parents.get(file) ?? []).some((p) => own.get(p)?.has(name)) ||
    (fallback && anywhere.has(name))
  const found: string[] = []
  for (const [file, text] of sources)
    for (const m of text.matchAll(/var\(\s*(--[a-zA-Z0-9_-]+)([$]?)\s*(,?)/g)) {
      const [, name, templated, fallback] = m
      if (templated || name.startsWith('--bits-')) continue
      if (!definedFor(file, name, fallback === ','))
        found.push(`${file}: ${name}`)
    }
  return found
}

it('every var(--x) names a defined custom property', () => {
  const sources = sourceFiles().map((f): [string, string] => [
    f,
    readFileSync(f, 'utf8'),
  ])
  const found = undefinedTokens(sources).map((u) => u.slice(UI.length + 1))
  expect(found).toEqual([])
})

it('a custom property is defined by :root, its own file or its parent', () => {
  const at = (name: string) => join(UI, 'probe', name)
  const found = undefinedTokens([
    [at('app.css'), ':root {\n  --bg: #fff;\n}\n.x { --stray: 1px }\n'],
    [
      at('Parent.svelte'),
      "<script>import Child from './Child.svelte'</script>\n<style>.p { --gap: 2px }</style>",
    ],
    [
      at('Child.svelte'),
      '<style>.c { margin: var(--gap); color: var(--bg); padding: var(--stray); }</style>',
    ],
    [
      at('Other.svelte'),
      // --gap is not this file's parent's; a fallback knob set somewhere is
      // fine, a templated name is skipped
      '<style>.o { margin: var(--gap); width: var(--stray, 1px); color: var(--tone-${t}); }</style>',
    ],
  ])
  expect(found).toEqual([
    `${at('Child.svelte')}: --stray`,
    `${at('Other.svelte')}: --gap`,
  ])
})

// app.css is near its cap; a shared class that would grow it goes in patterns.css
it('app.css stays at or under 600 lines', () => {
  const lines = readFileSync(join(UI, 'src/app.css'), 'utf8').split('\n')
  // a trailing newline is not a line
  const count = lines.at(-1) === '' ? lines.length - 1 : lines.length
  expect(count).toBeLessThanOrEqual(600)
})

// A popover or menu opened inside the viewer or a modal shows on top: it
// closes on a press outside, so it is never left over a modal opened later.
// A question asked from a sheet sits over that sheet
it('the layers stack page, modal, question, popover', () => {
  const css = readFileSync(join(UI, 'src/app.css'), 'utf8')
  const layer = (name: string) =>
    Number(css.match(new RegExp(`--layer-${name}:\\s*(\\d+)`))?.[1])
  const order = ['sticky', 'drawer', 'modal', 'ask', 'popover'].map(layer)
  expect(order.every((n) => Number.isFinite(n))).toBe(true)
  expect([...order].sort((a, b) => a - b)).toEqual(order)
  expect(new Set(order).size).toBe(order.length)
})
