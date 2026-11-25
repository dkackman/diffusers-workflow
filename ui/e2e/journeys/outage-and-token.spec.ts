import { expect, test, type Page } from '@playwright/test'

// The journeys of the resilience phase, end to end against the real server:
// an outage and its recovery, a locked server and a token, a job page whose
// event stream drops, a finish toast with Open, and a double-clicked Run
// again. The outage and the lock are made in the browser (page.route on
// /api/**), the way a dropped network or a server with --token looks to the
// page; scratch state is read through the HTTP API, never from files.
const TOKEN = 'e2e-token'
const WF = 'e2e-card'
const WS = 'e2e-sidebar'

type Request = Page['request']

async function jobIds(request: Request): Promise<string[]> {
  const read = await request.get('/api/jobs')
  expect(read.status()).toBe(200)
  const body = await read.json()
  const jobs: Array<{ id: string }> = Array.isArray(body) ? body : body.jobs
  return jobs.map((job) => job.id)
}

async function queueCard(request: Request): Promise<string> {
  const queued = await request.post('/api/jobs', {
    data: { workflow_path: WF },
  })
  expect(queued.ok()).toBe(true)
  const { id } = await queued.json()
  await expect
    .poll(
      async () => (await (await request.get(`/api/jobs/${id}`)).json()).status,
    )
    .toBe('succeeded')
  return id
}

test('an outage shows the offline banner and keeps the rows; recovery reloads them without a navigation', async ({
  page,
}) => {
  await page.goto('/#/prompts')
  const rows = page.locator('.card')
  await expect(rows.first()).toBeVisible()
  const before = await rows.count()
  expect(before).toBeGreaterThan(0)

  let navigated = 0
  page.on('framenavigated', () => navigated++)
  await page.route('**/api/**', (route) => route.abort())
  const banner = page.getByText(
    "Can't reach the server. Retrying every few seconds.",
  )
  // within one status poll (STATUS_POLL_MS, 5 s)
  await expect(banner).toBeVisible({ timeout: 10_000 })
  await expect(page.getByRole('button', { name: 'Retry now' })).toBeVisible()
  expect(await rows.count()).toBe(before)

  await page.unroute('**/api/**')
  await expect(banner).toHaveCount(0, { timeout: 15_000 })
  await expect(rows.first()).toBeVisible()
  expect(await rows.count()).toBe(before)
  expect(navigated).toBe(0)
})

test('a locked server shows the locked banner; a wrong token is refused, the right one fills the pages with no reload', async ({
  page,
  request,
}) => {
  // a second workspace, so the sidebar's list has something only a read shows
  const created = await request.post('/api/workspaces', { data: { name: WS } })
  expect([201, 409]).toContain(created.status())
  await page.route('**/api/**', (route) => {
    if (route.request().headers()['authorization'] === `Bearer ${TOKEN}`)
      return route.continue()
    return route.fulfill({
      status: 401,
      json: { detail: { code: 'unauthorized', message: 'API token required' } },
    })
  })
  await page.goto('/#/ws/default/workflows')
  await expect(page.getByText('This server needs an API token.')).toBeVisible({
    timeout: 10_000,
  })
  await expect(
    page.getByRole('button', { name: 'server & worker status' }),
  ).toHaveText('locked')

  let navigated = 0
  page.on('framenavigated', () => navigated++)
  const group = page.getByRole('group', { name: 'workspaces' })
  await expect(group).not.toContainText(WS)
  await page.getByRole('button', { name: 'Enter token' }).click()
  const input = page.getByPlaceholder('API token')
  await input.fill('wrong-token')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(page.getByText('The server refused that token')).toBeVisible()

  await input.fill(TOKEN)
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(page.getByText('Token accepted')).toBeVisible()
  await expect(page.getByText('This server needs an API token.')).toHaveCount(0)
  await expect(page.getByRole('link', { name: WF }).first()).toBeVisible({
    timeout: 15_000,
  })
  await expect(group).toContainText(WS)
  expect(navigated).toBe(0)
})

test('a job page whose event stream is aborted leaves running for the recorded state', async ({
  page,
  request,
}) => {
  const id = await queueCard(request)
  const recorded = await (await request.get(`/api/jobs/${id}`)).json()
  let first = true
  await page.route(
    (url) => url.pathname === `/api/jobs/${id}`,
    (route) => {
      if (!first) return route.continue()
      first = false
      const rest = { ...recorded, finished_at: undefined }
      return route.fulfill({
        json: { ...rest, status: 'running', historical: false },
      })
    },
  )
  await page.route(`**/api/jobs/${id}/events*`, (route) => route.abort())
  await page.goto(`/#/ws/default/jobs/${id}`)
  // the stream drops, the record is read again and says what happened
  await expect(page.locator('.chip.succeeded')).toBeVisible({ timeout: 20_000 })
  await expect(page.locator('.chip.running')).toHaveCount(0)
})

test('a run from its workflow page raises a finish toast whose Open goes to the job', async ({
  page,
  request,
}) => {
  const known = new Set(await jobIds(request))
  await page.goto(`/#/ws/default/workflows/${WF}`)
  await page.getByRole('button', { name: 'Run', exact: true }).click()
  // Run lands on the job's own page, which already shows its outcome: the
  // toast is for a job finishing while the person is elsewhere
  await expect(page).toHaveURL(/\/jobs\/[0-9a-f]+$/)
  await page.goto('/#/ws/default/workflows')
  const toast = page.locator('[data-sonner-toast]', { hasText: /succeeded/ })
  await expect(toast).toBeVisible({ timeout: 30_000 })
  await toast.getByRole('button', { name: 'Open' }).click()
  const fresh = (await jobIds(request)).filter((id) => !known.has(id))
  expect(fresh).toHaveLength(1)
  await expect(page).toHaveURL(new RegExp(`/jobs/${fresh[0]}$`))
})

test('a double-clicked Run again queues exactly one new job', async ({
  page,
  request,
}) => {
  const id = await queueCard(request)
  const known = new Set(await jobIds(request))
  await page.goto(`/#/ws/default/jobs/${id}`)
  const again = page.getByRole('button', { name: 'Run again', exact: true })
  await expect(again).toBeVisible()
  let posts = 0
  page.on('request', (r) => {
    if (r.method() === 'POST' && r.url().endsWith('/rerun')) posts++
  })
  await again.dblclick()
  await expect
    .poll(
      async () => (await jobIds(request)).filter((j) => !known.has(j)).length,
    )
    .toBeGreaterThanOrEqual(1)
  // The page has moved on to the new job: both clicks have been handled, and
  // a second POST would have left by now
  await expect(page).not.toHaveURL(new RegExp(`/jobs/${id}$`))
  expect(posts).toBe(1)
  expect((await jobIds(request)).filter((j) => !known.has(j))).toHaveLength(1)
})
