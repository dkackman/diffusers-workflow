import {
  expect,
  test,
  type APIRequestContext,
  type Page,
} from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'

/* Hostile content, a real server and a real browser.
 *
 * Everything the UI renders about a workflow, a prompt, a file or a job is
 * text somebody else chose - an MCP agent authoring a workflow, a file name
 * on disk, the output of a run. The UI keeps the API token in localStorage,
 * so a script that runs on this origin can read it. These specs plant
 * payloads in every field an untrusted author controls, walk the pages
 * that show them, and fail if any payload executes.
 *
 * Execution is detected, not inferred: every payload sets
 * `document.documentElement.dataset.xss` (and an alert would show up as a
 * dialog), and each page check first waits for the payload to be *visible
 * as text* - so a page that simply failed to render cannot pass.
 *
 * Content lives in its own workspace (e2e-xss) and one shared prompt, both
 * removed in afterAll, so the other specs never see it. Files on disk go
 * into the workspace directory the server reports - the spec runs on the
 * same machine as the fixture server. */

const WS = 'e2e-xss'
const PROMPT = 'e2e-xss-probe'
const MARK = (id: string) => `document.documentElement.dataset.xss='${id}'`

// For JSON fields: every shape an HTML sink would execute
const PAYLOAD = (id: string) =>
  `"'><img src=x onerror="${MARK(id)}"><svg onload="${MARK(id)}"></svg>` +
  `<script>${MARK(id)}</script>javascript:${MARK(id)}`
// For file names: no '/', so no closing tags
const FILE_PAYLOAD = `x"'><img src=x onerror="${MARK('file')}">`

// A 1x1 PNG, the same bytes serve_fixture.py writes
const PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM' +
    'IQAAAABJRU5ErkJggg==',
  'base64',
)

const hostileWorkflow = (id: string) => ({
  id,
  description: PAYLOAD('description'),
  variables: { note: PAYLOAD('default') },
  steps: [
    {
      name: 'compose',
      task: {
        command: 'compose_text',
        arguments: { parts: ['variable:note', PAYLOAD('argument')] },
      },
      result: { content_type: 'text/plain' },
    },
  ],
})

let outputsDir = ''
let assetsDir = ''
let htmlOutput = ''
let htmlJobId = ''

async function waitForJob(request: APIRequestContext, id: string) {
  for (let i = 0; i < 240; i++) {
    const job = await (await request.get(`/api/jobs/${id}`)).json()
    if (['succeeded', 'failed', 'cancelled'].includes(job.status)) return job
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
  throw new Error(`job ${id} did not finish`)
}

test.beforeAll(async ({ request }) => {
  // the first job spawns the worker, which imports torch
  test.setTimeout(180_000)
  const created = await request.post('/api/workspaces', { data: { name: WS } })
  expect([201, 409]).toContain(created.status())
  const server = await (await request.get(`/api/server?workspace=${WS}`)).json()
  outputsDir = server.directories.outputs
  assetsDir = server.directories.assets
  expect(outputsDir).toContain(WS)

  // names the server itself will list: gallery files and assets
  const outputs = path.join(outputsDir, 'hostile')
  fs.mkdirSync(outputs, { recursive: true })
  fs.writeFileSync(path.join(outputs, `${FILE_PAYLOAD}.png`), PNG)
  fs.writeFileSync(path.join(outputs, 'note.txt'), PAYLOAD('text-output'))
  const assets = assetsDir
  fs.mkdirSync(assets, { recursive: true })
  fs.writeFileSync(path.join(assets, `${FILE_PAYLOAD}.png`), PNG)

  // what an MCP agent can author
  const saved = await request.put(`/api/workflows/hostile?workspace=${WS}`, {
    data: { workflow: hostileWorkflow('hostile') },
  })
  expect(saved.ok()).toBeTruthy()
  const prompt = await request.put(`/api/prompts/${PROMPT}`, {
    data: {
      prompt: {
        text: PAYLOAD('prompt-text'),
        description: PAYLOAD('prompt-description'),
        tags: [PAYLOAD('prompt-tag').slice(0, 60)],
      },
    },
  })
  expect(prompt.ok()).toBeTruthy()

  // a job whose run wrote text/html - a content type the engine accepts
  const job = await request.post(`/api/jobs?workspace=${WS}`, {
    data: {
      workflow: {
        id: 'hostile-html',
        steps: [
          {
            name: 'page',
            task: {
              command: 'compose_text',
              arguments: {
                parts: [
                  `<script>document.title='stolen:'+localStorage.getItem('dw-api-token');${MARK('html-output')}</script>`,
                ],
              },
            },
            result: { content_type: 'text/html' },
          },
        ],
      },
    },
  })
  expect(job.ok()).toBeTruthy()
  htmlJobId = (await job.json()).id
  const finished = await waitForJob(request, htmlJobId)
  expect(finished.status).toBe('succeeded')
  const written = fs
    .readdirSync(outputsDir, {
      recursive: true,
    })
    .map(String)
    .find((name) => name.endsWith('.html'))
  expect(written).toBeTruthy()
  htmlOutput = written!
})

test.afterAll(async ({ request }) => {
  await request.delete(`/api/prompts/${PROMPT}`)
  await request.delete(`/api/workspaces/${WS}?acknowledged=true`)
})

/** Open a page and fail if anything planted ran. `visible` is text the
 * page must show first, so an empty page is a failure, not a pass. */
async function assertInert(page: Page, hash: string, visible: RegExp) {
  const dialogs: string[] = []
  page.on('dialog', async (dialog) => {
    dialogs.push(dialog.message())
    await dialog.dismiss()
  })
  await page.goto(hash)
  await expect(page.getByText(visible).first()).toBeVisible({
    timeout: 20_000,
  })
  // give onerror/onload handlers of broken images a chance to fire
  await page.waitForLoadState('networkidle')
  await page.waitForTimeout(300)
  const fired = await page.evaluate(
    () => document.documentElement.dataset.xss ?? null,
  )
  expect(fired, `payload executed on ${hash}`).toBeNull()
  expect(dialogs).toEqual([])
  // nothing the payload spelled became a live element or link
  expect(await page.locator('img[src="x"]').count()).toBe(0)
  expect(await page.locator('[onerror], [onload]').count()).toBe(0)
  expect(await page.locator('a[href^="javascript:" i]').count()).toBe(0)
}

// Each page, and the literal text it has to show to prove it rendered
const PAGES: [string, string, RegExp][] = [
  ['workflow catalog', `/#/ws/${WS}/workflows`, /onerror=/],
  ['workflow page', `/#/ws/${WS}/workflows/hostile`, /onerror=/],
  ['gallery', `/#/ws/${WS}/gallery`, /onerror=/],
  ['assets', `/#/ws/${WS}/assets`, /onerror=/],
  ['prompt library', '/#/shared/prompts', /e2e-xss-probe/],
]

for (const [label, hash, visible] of PAGES) {
  test(`hostile content on the ${label} stays text`, async ({ page }) => {
    await assertInert(page, hash, visible)
  })
}

test('hostile content in the prompt editor stays text', async ({ page }) => {
  // the editor shows the text inside form fields, not as page text
  await assertInert(page, `/#/shared/prompt-edit/${PROMPT}`, /e2e-xss-probe/)
  await expect(
    page.getByRole('textbox', { name: 'text', exact: true }),
  ).toHaveValue(/onerror=/)
})

test('hostile content in the workflow editor stays text', async ({ page }) => {
  test.setTimeout(60_000)
  await assertInert(page, `/#/ws/${WS}/edit/hostile`, /hostile/)
})

test('the job page lists the html output as an inert link', async ({
  page,
}) => {
  await assertInert(page, `/#/ws/${WS}/jobs/${htmlJobId}`, /hostile-html/)
  await expect(page.locator('a.filelink').first()).toBeVisible()
})

test('a gallery file with a hostile name opens inert', async ({ page }) => {
  await assertInert(page, `/#/ws/${WS}/gallery`, /onerror=/)
  await page
    .getByText(/onerror=/)
    .first()
    .click()
  await page.waitForTimeout(500)
  expect(
    await page.evaluate(() => document.documentElement.dataset.xss ?? null),
  ).toBeNull()
})

test.describe('a run that writes text/html', () => {
  // test.fail: the strict-xfail of Playwright - it fails the suite if this
  // starts passing, so a fix is noticed and the marker removed
  test.fail(
    true,
    'finding: a workflow with result content_type "text/html" writes an ' +
      '.html output that /outputs serves as text/html on the UI origin, ' +
      'where its script reads the API token from localStorage',
  )

  test('opening the output does not run it on the UI origin', async ({
    page,
  }) => {
    await page.goto('/')
    await page.evaluate(() =>
      localStorage.setItem('dw-api-token', JSON.stringify('secret-probe')),
    )
    await page.goto(`/outputs/${htmlOutput}?workspace=${WS}`)
    await page.waitForLoadState('load')
    const title = await page.title()
    const fired = await page.evaluate(
      () => document.documentElement.dataset.xss ?? null,
    )
    expect(title).not.toContain('secret-probe')
    expect(fired).toBeNull()
  })
})
