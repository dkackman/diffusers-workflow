// @vitest-environment node
// Where the --live state colour and the --select place colour may appear,
// held mechanically (ui/CLAUDE.md; app.css's header says what each means)
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, it } from 'vitest'
import { sourceFiles } from './arch-metrics.mjs'

const UI = join(dirname(fileURLToPath(import.meta.url)), '..')

// The places --live may appear: the header's running link and VRAM
// pressure, a running row and status chip (app.css: `.row-live`,
// `.chip.running`), the bar a changed tile or card shows (frames.css), the
// job page's progress bar and current step, the flow view's active step and
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
  'src/lib/pages/OverviewPage.svelte',
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

// One look for a chosen segment and one for a chosen chip: the selected
// styles live in Segmented.svelte and patterns.css, so no other component
// restyles `aria-pressed` or `.activebtn`
it('the segmented and chip selected styles live only in Segmented.svelte and patterns.css', () => {
  const declaring = sourceFiles()
    .filter((f) => /aria-pressed|\.activebtn/.test(styleOf(f)))
    .map((f) => f.slice(UI.length + 1).replaceAll('\\', '/'))
  expect(declaring.filter((f) => f !== 'src/lib/ui/Segmented.svelte')).toEqual(
    declaring.filter((f) => f === 'src/patterns.css'),
  )
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

// A var(--x) that names nothing silently falls back (or to nothing): every
// custom property read is defined by `--x:` somewhere in src, set with
// `style:--x`, or is a Bits UI `--bits-*` name
it('every var(--x) names a defined custom property', () => {
  const files = sourceFiles()
  const texts = files.map((f) => readFileSync(f, 'utf8'))
  const defined = new Set<string>()
  for (const t of texts) {
    for (const m of t.matchAll(/(?<![\w-])(--[a-zA-Z0-9_-]+)\s*:/g))
      defined.add(m[1])
    for (const m of t.matchAll(/style:(--[a-zA-Z0-9_-]+)/g)) defined.add(m[1])
  }
  const undefinedUses: string[] = []
  files.forEach((f, i) => {
    for (const m of texts[i].matchAll(/var\(\s*(--[a-zA-Z0-9_-]+)/g)) {
      if (!defined.has(m[1]) && !m[1].startsWith('--bits-'))
        undefinedUses.push(`${f.slice(UI.length + 1)}: ${m[1]}`)
    }
  })
  expect(undefinedUses).toEqual([])
})

// app.css is full; a shared class that would grow it goes in patterns.css
it('app.css stays at or under 600 lines', () => {
  const lines = readFileSync(join(UI, 'src/app.css'), 'utf8').split('\n')
  // a trailing newline is not a line
  const count = lines.at(-1) === '' ? lines.length - 1 : lines.length
  expect(count).toBeLessThanOrEqual(600)
})
