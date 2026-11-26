import { expect, test, type Locator, type Page } from '@playwright/test'
import type {
  EventLogReply,
  JobDetail,
  JobEvent,
  JobWorkflow,
} from '../src/lib/types'

// The job page's Event log (#1478), on a mocked finished job whose history
// kept only the tail of its log: the section opens on a click, reads the
// journal route once, and shows the server's note above the rows.
// Screenshots of the section in both themes land in this spec's
// test-results folder.

const ID = 'j-eventlog'
const NOTE =
  'Only the last 200 events of this job were retained; everything before seq 37 was dropped when the job was recorded.'

const job: JobDetail = {
  id: ID,
  status: 'succeeded',
  workflow: 'text-to-image',
  workflow_name: 'text-to-image',
  workspace: 'default',
  arguments: {},
  acknowledged: 'none',
  acknowledged_cost: null,
  created_at: 1_700_000_000,
  started_at: 1_700_000_001,
  finished_at: 1_700_000_042,
  error: null,
  traceback: null,
  warnings: [],
  event_count: 4,
  manifest: [],
  run_dir: null,
  run_id: null,
  run_version: null,
  historical: true,
}

const events: JobEvent[] = [
  { seq: 37, event: 'step_start', at: 1.2, step: 'generate' },
  {
    seq: 38,
    event: 'log',
    at: 3.4,
    message: 'loading pipeline StableDiffusionPipeline',
  },
  {
    seq: 39,
    event: 'warning',
    kind: 'phase_stall',
    at: 30.1,
    message: 'no progress for 20s in denoise',
  },
  { seq: 40, event: 'step_end', at: 40.9, step: 'generate', files: [] },
]

async function mockJob(page: Page) {
  const logCalls: string[] = []
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
          definition: { id: 'text-to-image', steps: [] },
          realized: true,
          seed_variable: null,
        } satisfies JobWorkflow,
      }),
  )
  await page.route(
    (url) => url.pathname === `/api/jobs/${ID}/journal`,
    (route) => {
      logCalls.push(new URL(route.request().url()).search)
      return route.fulfill({
        json: {
          id: ID,
          status: 'succeeded',
          events,
          last_seq: 40,
          truncated: false,
          note: NOTE,
        } satisfies EventLogReply,
      })
    },
  )
  return logCalls
}

// EasyPrivacy's generic rule `/event-log?`, which uBlock Origin, AdGuard and
// the like apply to this server's own requests (#1731)
test('the log reads past a content blocker that drops /event-log?', async ({
  page,
}) => {
  await mockJob(page)
  await page.route(/\/event-log\?/, (route) => route.abort('blockedbyclient'))
  await page.goto(`/#/ws/default/jobs/${ID}`)
  const section = page.getByRole('region', { name: 'Event log' })
  await section.getByRole('button', { name: /Event log/ }).click()
  await expect(section.locator('tbody tr')).toHaveCount(4)
})

// The same read lost on its own: the server answers, so it is not an outage
test('one lost read is not reported as the server being down', async ({
  page,
}) => {
  await mockJob(page)
  await page.route(
    (url) => url.pathname === `/api/jobs/${ID}/journal`,
    (route) => route.abort('blockedbyclient'),
  )
  await page.goto(`/#/ws/default/jobs/${ID}`)
  const section = page.getByRole('region', { name: 'Event log' })
  await section.getByRole('button', { name: /Event log/ }).click()
  await expect(section).toContainText("This request didn't get through")
  await expect(section.getByRole('button', { name: 'Retry' })).toBeVisible()
  await expect(page.getByText("Can't reach the server")).toHaveCount(0)
})

for (const theme of ['light', 'dark'] as const) {
  test(`event log on a finished job with a trimmed log (${theme})`, async ({
    page,
  }, testInfo) => {
    await page.emulateMedia({ colorScheme: theme })
    const logCalls = await mockJob(page)
    await page.goto(`/#/ws/default/jobs/${ID}`)

    const section = page.getByRole('region', { name: 'Event log' })
    const toggle = section.getByRole('button', { name: /Event log/ })
    await expect(toggle).toHaveAttribute('aria-expanded', 'false')
    // Collapsed, it has asked for nothing
    expect(logCalls).toEqual([])

    await toggle.click()
    await expect(section.locator('tbody tr')).toHaveCount(4)
    await expect(section).toContainText('everything before seq 37 was dropped')
    await expect(section.locator('tbody tr').nth(2)).toContainText(
      'warning · phase_stall',
    )
    // A finished job's log does not grow
    await expect(
      section.getByRole('button', { name: 'Refresh event log' }),
    ).toHaveCount(0)
    expect(logCalls.length).toBe(1)
    expect(logCalls[0]).toContain('after=-1')
    expect(logCalls[0]).toContain('limit=500')

    await section.screenshot({
      path: testInfo.outputPath(`job-event-log-${theme}.png`),
    })
  })
}

// A running job's log (#1499), from a real live answer on mini-ai: a memory
// reading's payload is long, and two of them once pushed every row after
// seq 11 below the box's fold, so Refresh's rows looked like they never
// came. Each row is one line, and a live log scrolls to its newest row.
const LIVE = 'j-eventlog-live'
const memory = (seq: number) => ({
  seq,
  at: 0.0,
  event: 'memory',
  info: {
    run_count: 3,
    gpu_available: true,
    gpu_memory_allocated_mb: 4102.5546875,
    gpu_memory_reserved_mb: 4161.6875,
    gpu_memory_free_mb: 51134.3125,
    gpu_device_name: 'Apple M5 Pro (MPS)',
    host_memory_rss_mb: 917.34375,
    host_memory_peak_rss_mb: 917.390625,
    host_memory_total_mb: 65536.0,
    host_memory_available_mb: 36195.3125,
    gpu_memory_total_mb: 55296.0,
    step_cache: {
      entries: 0,
      max_entries: 128,
      retained_bytes: 0,
      max_retained_bytes: 4294967296,
    },
    host_memory_job_peak_rss_mb: null,
  },
})
const step = (i: number) => ({
  seq: 12 + i,
  at: 2.8 * i,
  event: 'pipeline_step',
  step: i,
  total_steps: 26,
})
const liveEvents: JobEvent[] = [
  { seq: 0, at: 0, event: 'job_status', status: 'queued' },
  { seq: 1, at: 0, event: 'job_status', status: 'running' },
  memory(2),
  { seq: 3, at: 0, event: 'log', message: 'text-to-image' },
  { seq: 4, at: 0, event: 'log', message: 'Executing workflow: text-to-image' },
  {
    seq: 5,
    at: 0,
    event: 'run_start',
    run_id: '20261119-171222-dc64dcbb',
    version: 3,
    identity: 'templates/text-to-image',
    run_dir: 'templates/text-to-image/20261119-171222-dc64dcbb',
  },
  {
    seq: 6,
    at: 0,
    event: 'workflow_start',
    workflow: 'text-to-image',
    total_steps: 1,
    steps: ['main'],
    seed: 7070643648549579,
  },
  {
    seq: 7,
    at: 0,
    event: 'step_start',
    workflow: 'text-to-image',
    step: 'main',
    index: 0,
    total_steps: 1,
  },
  {
    seq: 8,
    at: 0,
    event: 'phase',
    phase: 'cached',
    detail: 'stable-diffusion-v1-5/stable-diffusion-v1-5',
  },
  memory(9),
  {
    seq: 10,
    at: 0,
    event: 'iteration_start',
    step: 'main',
    iteration: 1,
    total_iterations: 1,
  },
  {
    seq: 11,
    at: 0,
    event: 'phase',
    phase: 'generating',
    detail: 'stable-diffusion-v1-5/stable-diffusion-v1-5',
  },
  memory(12),
  ...Array.from({ length: 12 }, (_, i) => step(i + 1)),
]

/** The row sits inside the log's scroll box, not clipped below its fold */
async function inBox(row: Locator): Promise<boolean> {
  return row.evaluate((tr) => {
    const box = tr.closest('#event-log-body')!.getBoundingClientRect()
    const r = tr.getBoundingClientRect()
    return r.top >= box.top - 1 && r.bottom <= box.bottom + 1
  })
}

test("a running job's log shows every row and grows on Refresh", async ({
  page,
}) => {
  const served = [...liveEvents]
  const afters: string[] = []
  await page.route(
    (url) => url.pathname === `/api/jobs/${LIVE}`,
    (route) =>
      route.fulfill({
        json: {
          ...job,
          id: LIVE,
          status: 'running',
          finished_at: null,
          event_count: served.length,
          historical: false,
        } satisfies JobDetail,
      }),
  )
  await page.route(
    (url) => url.pathname === `/api/jobs/${LIVE}/workflow`,
    (route) =>
      route.fulfill({
        json: {
          id: LIVE,
          definition: { id: 'text-to-image', steps: [] },
          realized: true,
          seed_variable: null,
        } satisfies JobWorkflow,
      }),
  )
  // The live stream is held open with nothing on it: the log reads its own route
  await page.route(
    (url) => url.pathname === `/api/jobs/${LIVE}/events`,
    (route) =>
      route.fulfill({
        headers: { 'content-type': 'text/event-stream' },
        body: ': idle\n\n',
      }),
  )
  await page.route(
    (url) => url.pathname === `/api/jobs/${LIVE}/journal`,
    (route) => {
      const after = new URL(route.request().url()).searchParams.get('after')!
      afters.push(after)
      const rows = served.filter((e) => e.seq > Number(after))
      return route.fulfill({
        json: {
          id: LIVE,
          status: 'running',
          events: rows,
          last_seq: rows.length ? rows[rows.length - 1].seq : Number(after),
          truncated: false,
          note: null,
        } satisfies EventLogReply,
      })
    },
  )
  await page.goto(`/#/ws/default/jobs/${LIVE}`)
  const section = page.getByRole('region', { name: 'Event log' })
  await section.getByRole('button', { name: /Event log/ }).click()
  const rows = section.locator('tbody tr')
  await expect(rows).toHaveCount(25)
  await expect(section).toContainText('25 events, seq 0–24')
  // The newest row is in view, not below the box's fold
  await expect(rows.last()).toContainText('pipeline_step')
  expect(await inBox(rows.last())).toBe(true)
  // A row for an event with no sentence of its own is its name and the
  // Event data disclosure: two lines at most, never a tall block
  const heights = await rows.evaluateAll((trs) =>
    trs.map((tr) => tr.getBoundingClientRect().height),
  )
  // Measured tallest/shortest row: 1.85 (25 rows); the bound is that + 10%
  expect(Math.max(...heights)).toBeLessThan(Math.min(...heights) * 2.03)

  served.push(step(13), step(14))
  await section.getByRole('button', { name: 'Refresh event log' }).click()
  await expect(rows).toHaveCount(27)
  expect(afters).toEqual(['-1', '24'])
  const seqs = await section
    .locator('tbody tr td:first-child')
    .allTextContents()
  expect(seqs).toEqual(Array.from({ length: 27 }, (_, i) => String(i)))
  expect(await inBox(rows.last())).toBe(true)
  await expect(section).toContainText('27 events, seq 0–26')
})
