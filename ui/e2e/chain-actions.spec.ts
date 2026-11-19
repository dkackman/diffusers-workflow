import { expect, test, type Page } from '@playwright/test'

// The chain segment table's actions (#1476), on a mocked finished H3 job
// that stopped after 2 of its 4 planned segments and kept both segment
// files: Repair this segment opens repair-segment with its fields filled,
// and Continue the chain posts the rerun that resumes from segment 2.

const ID = 'j-chain-actions'
const NEXT = 'j-chain-actions-next'
const H3 = 'MiniMaxAI/MiniMax-H3'
const REPAIR = 'templates/minimax/repair-segment'

const shots = [1, 2].map((i) => ({
  name: `segment ${i}`,
  start_frame: (i - 1) * 121,
  num_frames: 121,
  planned_segments: 4,
  attempts: 1,
  segment_file: `take.segment-00${i}.mp4`,
}))

const job = (id: string) => ({
  id,
  status: 'succeeded',
  workflow: 'narrated-take',
  workflow_name: 'narrated-take',
  workspace: 'default',
  arguments: { stop_after: 2, keep_segments: true },
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
    {
      step: 'take',
      files: [
        'run/take.mp4',
        'run/take.segment-001.mp4',
        'run/take.segment-002.mp4',
      ],
      shots,
    },
  ],
  run_dir: null,
  run_id: null,
  run_version: null,
  historical: true,
})

const definition = {
  id: 'narrated-take',
  variables: { stop_after: null, resume_from: null, keep_segments: false },
  steps: [
    {
      name: 'take',
      pipeline: {
        from_pretrained_arguments: { model_name: H3 },
        chain: {
          save_segments: true,
          keep_segments: 'variable:keep_segments',
          stop_after: 'variable:stop_after',
          resume_from: 'variable:resume_from',
        },
        arguments: {},
      },
    },
  ],
}

const repairTemplate = {
  id: 'repair-segment',
  variables: { chain_take: '', chain_output: '', segment: 2 },
  steps: [
    {
      name: 'repair',
      pipeline: {
        from_pretrained_arguments: { model_name: H3 },
        arguments: {},
      },
    },
  ],
}

async function mockJob(page: Page) {
  for (const id of [ID, NEXT]) {
    await page.route(
      (url) => url.pathname === `/api/jobs/${id}`,
      (route) => route.fulfill({ json: job(id) }),
    )
    await page.route(
      (url) => url.pathname === `/api/jobs/${id}/workflow`,
      (route) =>
        route.fulfill({
          json: { id, definition, realized: true, seed_variable: null },
        }),
    )
  }
  await page.route(
    (url) => url.pathname === `/api/workflows/${REPAIR}`,
    (route) => route.fulfill({ json: repairTemplate }),
  )
  await page.route('**/outputs/**', (route) => route.fulfill({ status: 404 }))
}

test('Repair this segment opens repair-segment with the take and segment filled', async ({
  page,
}) => {
  await mockJob(page)
  let submitted = false
  await page.route(
    (url) => url.pathname === '/api/jobs',
    (route) => {
      if (route.request().method() === 'POST') submitted = true
      return route.fallback()
    },
  )
  await page.goto(`/#/ws/default/jobs/${ID}`)

  const table = page.getByRole('region', { name: 'Chain segments: take' })
  const rows = table.locator('tbody tr')
  await expect(rows).toHaveCount(2)
  const first = rows.nth(0).getByRole('button', { name: 'Repair this segment' })
  await expect(first).toBeDisabled()
  await expect(rows.nth(0)).toContainText(
    'segment 1 cannot be repaired - re-render with a new seed',
  )

  const second = rows
    .nth(1)
    .getByRole('button', { name: 'Repair this segment' })
  await expect(second).toBeEnabled()
  await second.click()

  await expect(page).toHaveURL(
    /#\/ws\/default\/workflows\/templates\/minimax\/repair-segment$/,
  )
  await expect(page.locator('#var-chain_take')).toHaveValue(
    'output:run/take.mp4',
  )
  await expect(page.locator('#var-segment')).toHaveValue('2')
  // Filled in, not run
  expect(submitted).toBe(false)
})

test('Continue the chain posts the rerun that resumes from the last segment', async ({
  page,
}) => {
  await mockJob(page)
  let body: unknown = null
  await page.route(
    (url) => url.pathname === `/api/jobs/${ID}/rerun`,
    (route) => {
      body = route.request().postDataJSON()
      return route.fulfill({ json: job(NEXT) })
    },
  )
  // The confirm prices the rerun with validate, as a run would be priced
  let priced: any = null
  await page.route('**/api/validate**', (route) => {
    priced = route.request().postDataJSON()
    return route.fulfill({
      json: {
        valid: true,
        errors: [],
        warnings: [],
        plan: {
          fingerprint: 'sha256:abc',
          steps: 1,
          list_entries: {},
          cached_steps: 0,
          elided_steps: [],
          downloads_required: [],
          estimate: {
            minutes: 12,
            basis: 'catalog',
            device: 'mps',
            measured_on: 'M4 Max',
            partial: false,
            unpriced: [],
            runs: null,
            cached_minutes: 12,
          },
          workspace: 'default',
        },
      },
    })
  })
  await page.goto(`/#/ws/default/jobs/${ID}`)

  const table = page.getByRole('region', { name: 'Chain segments: take' })
  await table.getByRole('button', { name: 'Continue the chain' }).click()
  const dialog = page.getByRole('alertdialog')
  await expect(dialog).toContainText('Estimate: ~12 min on mps')
  expect(priced).toMatchObject({
    workflow_path: 'narrated-take',
    workspace: 'default',
    arguments: {
      resume_from: 'output:run/take.segment-002.mp4',
      stop_after: null,
    },
  })
  // Pricing it queued nothing
  expect(body).toBeNull()
  await dialog.getByRole('button', { name: 'Continue' }).click()

  await expect(page).toHaveURL(new RegExp(`#/ws/default/jobs/${NEXT}$`))
  expect(body).toEqual({
    new_seed: false,
    arguments: {
      resume_from: 'output:run/take.segment-002.mp4',
      stop_after: null,
    },
  })
})
