import { expect, test, type Page } from '@playwright/test'

// The job page's Event log (#1478), on a mocked finished job whose history
// kept only the tail of its log: the section opens on a click, reads the
// event-log route once, and shows the server's note above the rows.
// Screenshots of the section in both themes land in this spec's
// test-results folder.

const ID = 'j-eventlog'
const NOTE =
  'Only the last 200 events of this job were retained; everything before seq 37 was dropped when the job was recorded.'

const job = {
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

const events = [
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
        },
      }),
  )
  await page.route(
    (url) => url.pathname === `/api/jobs/${ID}/event-log`,
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
        },
      })
    },
  )
  return logCalls
}

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
    await expect(section.getByRole('button', { name: 'Refresh' })).toHaveCount(
      0,
    )
    expect(logCalls.length).toBe(1)
    expect(logCalls[0]).toContain('after=-1')
    expect(logCalls[0]).toContain('limit=500')

    await section.screenshot({
      path: testInfo.outputPath(`job-event-log-${theme}.png`),
    })
  })
}
