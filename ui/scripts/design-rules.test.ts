// @vitest-environment node
// Where the --live state colour may appear, held mechanically (ui/CLAUDE.md;
// app.css's header says why --live means machine state and nothing else)
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, it } from 'vitest'
import { sourceFiles } from './arch-metrics.mjs'

const UI = join(dirname(fileURLToPath(import.meta.url)), '..')

// The places --live may appear: the header's running link and VRAM
// pressure, a running job's row and status chip, the job page's progress
// bar and current step, the flow view's active step, a model download in
// flight, and the focus ring
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
