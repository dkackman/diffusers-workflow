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
// pressure, a running job's row and status chip, the job page's progress
// bar and current step, the flow view's active step, and a model download
// in flight
const LIVE_FILES = [
  'src/App.svelte',
  'src/app.css',
  'src/lib/editor/FlowView.svelte',
  'src/lib/job/JobProgress.svelte',
  'src/lib/pages/JobsPage.svelte',
  'src/lib/pages/ModelsPage.svelte',
  'src/lib/pages/OverviewPage.svelte',
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
