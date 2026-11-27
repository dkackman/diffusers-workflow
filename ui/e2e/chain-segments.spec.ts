import { expect, test, type Page } from './test'
import type { JobDetail, JobWorkflow, ManifestEntry } from '../src/lib/types'

type Shots = NonNullable<ManifestEntry['shots']>

// The job page's chain segment table (#1475), on a mocked finished job:
// a checked chain stopped short of its plan (`stop_after`), with a resumed
// segment, a segment that passed, one that kept a re-roll and still failed
// a seam measure, and one with a warn-only texture warning; beside it a
// chain recorded before measures were kept, and a step that is no chain.
// Screenshots of the table in both themes land in this spec's test-results
// folder. The fixture has no video files, so the thumbnails are empty
// frames here; their sources are asserted instead.

const ID = 'j-chain-segments'

const m = (
  value: number,
  ceiling: number | null,
  extra: { bound?: 'max' | 'min'; warn_only?: boolean; failed?: boolean } = {},
) => ({
  value,
  ceiling,
  bound: 'max' as const,
  warn_only: false,
  failed: false,
  ...extra,
})

const shots: Shots = [
  {
    name: 'segment 1',
    start_frame: 0,
    num_frames: 121,
    planned_segments: 6,
    resumed: true,
  },
  {
    name: 'segment 2',
    start_frame: 121,
    num_frames: 121,
    planned_segments: 6,
    attempts: 1,
    measures: {
      jump_ratio: m(0.41, 0.6),
      look_ratio: m(0.22, 0.5),
      texture_ratio: m(1.08, 1.5, { warn_only: true }),
      audio_dbfs: m(-21.4, -70, { bound: 'min' }),
    },
  },
  {
    name: 'segment 3',
    start_frame: 242,
    num_frames: 121,
    planned_segments: 6,
    attempts: 3,
    seed: 2000049,
    check_failed: ['jump_ratio'],
    measures: {
      jump_ratio: m(0.83, 0.6, { failed: true }),
      look_ratio: m(0.31, 0.5),
      texture_ratio: m(1.12, 1.5, { warn_only: true }),
      audio_dbfs: m(-19.9, -70, { bound: 'min' }),
    },
  },
  {
    name: 'segment 4',
    start_frame: 363,
    num_frames: 121,
    planned_segments: 6,
    attempts: 1,
    check_failed: ['texture_ratio'],
    segment_file: 'take.segment-004.mp4',
    measures: {
      jump_ratio: m(0.38, 0.6),
      look_ratio: m(0.27, 0.5),
      texture_ratio: m(1.71, 1.5, { warn_only: true, failed: true }),
      audio_dbfs: m(-20.6, -70, { bound: 'min' }),
    },
  },
]

const oldShots = [0, 1].map((i) => ({
  name: `segment ${i + 1}`,
  start_frame: i * 97,
  num_frames: 97,
  planned_segments: 2,
  attempts: 1,
}))

const job: JobDetail = {
  id: ID,
  status: 'succeeded',
  workflow: 'chained-shot',
  workflow_name: 'chained-shot',
  workspace: 'default',
  arguments: { retries: 2 },
  acknowledged: 'none',
  acknowledged_cost: null,
  created_at: 1_700_000_000,
  started_at: 1_700_000_001,
  finished_at: 1_700_000_942,
  error: null,
  traceback: null,
  warnings: [],
  event_count: 0,
  manifest: [
    { step: 'still', files: ['run/still.png'] },
    {
      step: 'take',
      files: ['run/take.mp4', 'run/take.segment-004.mp4'],
      shots,
    },
    { step: 'old_take', files: ['run/old_take.mp4'], shots: oldShots },
  ],
  run_dir: null,
  run_id: null,
  run_version: null,
  historical: true,
}

const definition = {
  id: 'chained-shot',
  variables: { frame_rate: 24, retries: 0 },
  steps: [
    { name: 'still', pipeline: { arguments: {} } },
    {
      name: 'take',
      pipeline: {
        chain: { check: true, retries: 'variable:retries', stop_after: 4 },
        arguments: { frame_rate: 'variable:frame_rate' },
      },
    },
    { name: 'old_take', pipeline: { chain: { check: true }, arguments: {} } },
  ],
}

async function mockJob(page: Page) {
  await page.route(
    (url) => url.pathname === `/api/jobs/${ID}`,
    (route) => route.fulfill({ json: job }),
  )
  await page.route(
    (url) => url.pathname === `/api/jobs/${ID}/workflow`,
    (route) =>
      route.fulfill({
        json: {
          id: ID,
          definition,
          realized: true,
          seed_variable: null,
        } satisfies JobWorkflow,
      }),
  )
  await page.route('**/outputs/**', (route) => route.fulfill({ status: 404 }))
}

for (const theme of ['light', 'dark'] as const) {
  test(`chain segment table on a finished chained job (${theme})`, async ({
    page,
  }, testInfo) => {
    await page.emulateMedia({ colorScheme: theme })
    await mockJob(page)
    await page.goto(`/#/ws/default/jobs/${ID}`)

    const table = page.getByRole('region', { name: 'Chain segments: take' })
    await expect(table).toContainText('4 of 6 planned')
    const rows = table.locator('tbody tr')
    await expect(rows).toHaveCount(4)

    await expect(rows.nth(0)).toContainText('resumed')
    await expect(rows.nth(0)).toContainText('not recorded')
    await expect(rows.nth(1).locator('.measure.pass')).toHaveCount(4)
    await expect(rows.nth(1)).toContainText('-21.4 ≥ -70')
    await expect(rows.nth(2)).toContainText('3 of 3')
    await expect(rows.nth(2)).toContainText('kept seed 2000049')
    await expect(rows.nth(2).locator('.measure.fail')).toContainText(
      '0.83 ≤ 0.6',
    )
    await expect(rows.nth(3).locator('.measure.warn')).toContainText(
      '1.71 ≤ 1.5',
    )
    await expect(
      table.getByRole('columnheader', { name: 'texture_ratio' }),
    ).toBeVisible()

    // The kept segment file plays itself; the others seek the take. A
    // thumbnail loads only once near the viewport
    await table.scrollIntoViewIfNeeded()
    await expect(rows.nth(3).locator('video')).toHaveAttribute(
      'src',
      /take\.segment-004\.mp4.*#t=0\.001$/,
    )
    await expect(rows.nth(2).locator('video')).toHaveAttribute(
      'src',
      /take\.mp4.*#t=10\.083$/,
    )

    const old = page.getByRole('region', { name: 'Chain segments: old_take' })
    await expect(old).toContainText('2 of 2 planned')
    await expect(old.locator('tbody tr').nth(1)).toContainText('not recorded')
    await expect(old).toContainText('1 of 1')

    // A step that is no chain gets no table
    await expect(
      page.getByRole('region', { name: 'Chain segments: still' }),
    ).toHaveCount(0)

    await table.screenshot({
      path: testInfo.outputPath(`chain-segments-${theme}.png`),
    })
  })
}
