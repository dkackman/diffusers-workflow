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
// `.chip.running`), the job page's progress bar and current step, the flow
// view's active step and segment and its legend's key to them, a model
// download in flight, and a request's progress bar
const LIVE_FILES = [
  'src/App.svelte',
  'src/app.css',
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
