import { expect, test, type Page } from '@playwright/test'

// The journeys of the data-safety phase, end to end against the real server:
// an edit is never lost to a fast save, a view switch, a link, a failed load,
// a second tab or an upload that shares a name. Each case owns a scratch
// workflow or prompt of its own (made and removed through the API) so the
// seeded files other specs read stay as the fixture made them. State is
// checked through the HTTP API, never by reading files.
const WF = 'e2e-safety'
const PROMPT = 'e2e-safety-prompt'
const EDIT = `/#/ws/default/edit/${WF}`
const BASE = {
  id: 'E2ESafety',
  description: 'Scratch workflow for the editor-safety journeys.',
  variables: { line: 'Title card' },
  steps: [
    {
      name: 'card',
      task: {
        command: 'compose_text',
        arguments: { parts: ['variable:line'] },
      },
      result: { content_type: 'text/plain' },
    },
  ],
}

type Request = Page['request']

async function putWorkflow(request: Request, workflow: object, name = WF) {
  const saved = await request.put(`/api/workflows/${name}`, {
    data: { workflow },
  })
  expect(saved.status()).toBe(200)
}

async function serverId(request: Request, name = WF): Promise<string> {
  const read = await request.get(`/api/workflows/${name}`)
  expect(read.status()).toBe(200)
  return (await read.json()).id
}

async function jsonView(page: Page) {
  await page.getByRole('button', { name: 'JSON', exact: true }).click()
  await expect(page.locator('.monaco-editor').first()).toBeVisible({
    timeout: 20_000,
  })
}

/** Replace the Monaco document with `text`, typed the way a person would */
async function typeJson(page: Page, text: string) {
  await page.locator('.view-lines').first().click()
  await page.keyboard.press('ControlOrMeta+a')
  await page.keyboard.press('Backspace')
  await page.keyboard.insertText(text)
}

test.beforeEach(async ({ request }) => {
  await putWorkflow(request, BASE)
  await request.delete(`/api/prompts/${PROMPT}`)
})

test.afterEach(async ({ request }) => {
  await request.delete(`/api/workflows/${WF}`)
  await request.delete(`/api/prompts/${PROMPT}`)
})

test('JSON view: a new id typed and saved at once survives a reload', async ({
  page,
  request,
}) => {
  await page.goto(EDIT)
  await jsonView(page)
  await typeJson(page, JSON.stringify({ ...BASE, id: 'TypedAtOnce' }))
  // no pause: the debounce has not fired, the save must flush the editor
  await page.keyboard.press('ControlOrMeta+s')
  await expect(page.getByText(`Saved to ${WF}`)).toBeVisible()
  expect(await serverId(request)).toBe('TypedAtOnce')
  await page.reload()
  await expect(page.locator('input.wfid')).toHaveValue('TypedAtOnce')
})

test('form view: a field typed in and saved at once, without a blur, is saved', async ({
  page,
  request,
}) => {
  await page.goto(EDIT)
  const line = page.locator('#wfvar-line')
  await expect(line).toHaveValue('Title card')
  // fill leaves focus in the field: its change has not fired
  await line.fill('Typed, still focused')
  await page.keyboard.press('ControlOrMeta+s')
  await expect(page.getByText(`Saved to ${WF}`)).toBeVisible()
  const read = await request.get(`/api/workflows/${WF}`)
  expect((await read.json()).variables.line).toBe('Typed, still focused')
  await expect(page.locator('.chip.unsaved')).toHaveCount(0)
})

test('form view: Back from a field still focused asks before dropping the edit', async ({
  page,
}) => {
  await page.goto('/#/ws/default/jobs')
  await page.goto(EDIT)
  const line = page.locator('#wfvar-line')
  await expect(line).toHaveValue('Title card')
  await line.fill('Not blurred')
  await page.goBack()
  const dialog = page.getByRole('alertdialog')
  await expect(dialog).toContainText(
    `Leave without saving your changes to ${WF}?`,
  )
  await dialog.getByRole('button', { name: 'Keep editing' }).click()
  await expect(page).toHaveURL(new RegExp(`/edit/${WF}$`))
  await expect(line).toHaveValue('Not blurred')
})

test('JSON view: switching to form shows the edit, and broken JSON keeps the view', async ({
  page,
}) => {
  await page.goto(EDIT)
  await jsonView(page)
  await typeJson(page, JSON.stringify({ ...BASE, id: 'ViaForm' }))
  await page.getByRole('button', { name: 'form', exact: true }).click()
  await expect(page.locator('input.wfid')).toHaveValue('ViaForm')

  await jsonView(page)
  await typeJson(page, '{"id": "Broken", ')
  await page.getByRole('button', { name: 'form', exact: true }).click()
  // the view stays on JSON, and says why
  await expect(page.locator('.monaco-editor').first()).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Variables' })).toHaveCount(0)
  await expect(
    page.getByText('Fix the JSON before switching view'),
  ).toBeVisible()
})

test('an unsaved edit is guarded against leaving: Keep editing stays, Leave goes, Back shows the saved version', async ({
  page,
}) => {
  await page.goto(EDIT)
  await expect(page.locator('input.wfid')).toHaveValue('E2ESafety')
  await page.locator('input.wfid').fill('Unsaved')
  await expect(page.locator('.chip.unsaved')).toBeVisible()

  const jobs = page.getByRole('link', { name: 'Jobs', exact: true })
  await jobs.click()
  const dialog = page.getByRole('alertdialog')
  await expect(dialog).toContainText(
    `Leave without saving your changes to ${WF}?`,
  )
  await dialog.getByRole('button', { name: 'Keep editing' }).click()
  await expect(dialog).toHaveCount(0)
  await expect(page).toHaveURL(new RegExp(`/edit/${WF}$`))
  await expect(page.locator('input.wfid')).toHaveValue('Unsaved')
  await expect(page.locator('.chip.unsaved')).toBeVisible()

  await jobs.click()
  await page
    .getByRole('alertdialog')
    .getByRole('button', { name: 'Leave' })
    .click()
  await expect(page).toHaveURL(/#\/ws\/default\/jobs$/)

  await page.goBack()
  await expect(page).toHaveURL(new RegExp(`/edit/${WF}$`))
  await expect(page.locator('input.wfid')).toHaveValue('E2ESafety')
  await expect(page.locator('.chip.unsaved')).toHaveCount(0)
})

test('a workflow that fails to load offers Retry, with Save disabled until it does', async ({
  page,
}) => {
  const pattern = `**/api/workflows/${WF}`
  await page.route(pattern, (route) =>
    route.request().method() === 'GET'
      ? route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'boom' }),
        })
      : route.continue(),
  )
  await page.goto(EDIT)
  const failed = page.getByRole('alert').filter({ hasText: "Couldn't open" })
  await expect(failed).toBeVisible()
  await expect(page.getByRole('button', { name: /^Save/ })).toBeDisabled()

  await page.unroute(pattern)
  await failed.getByRole('button', { name: 'Retry' }).click()
  await expect(page.locator('input.wfid')).toHaveValue('E2ESafety')
  await expect(failed).toHaveCount(0)
  await expect(page.getByRole('button', { name: /^Save/ })).toBeEnabled()
})

test('saving a new workflow over an existing name asks to Replace, and Cancel leaves the file', async ({
  page,
  request,
}) => {
  await page.goto('/#/ws/default/edit')
  await jsonView(page)
  await typeJson(page, JSON.stringify({ ...BASE, id: 'Imposter' }))
  await page.getByRole('button', { name: 'form', exact: true }).click()
  await page.locator('#doc-savename').fill(WF)
  await page.getByRole('button', { name: /^Save/ }).click()
  const dialog = page.getByRole('alertdialog')
  await expect(dialog).toContainText(`${WF} already exists. Replace it?`)
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  await expect(dialog).toHaveCount(0)
  expect(await serverId(request)).toBe('E2ESafety')
})

test('two pages on one workflow: the second save is refused as stale', async ({
  context,
  request,
}) => {
  const a = await context.newPage()
  const b = await context.newPage()
  await a.goto(EDIT)
  await b.goto(EDIT)
  await expect(a.locator('input.wfid')).toHaveValue('E2ESafety')
  await expect(b.locator('input.wfid')).toHaveValue('E2ESafety')

  await a.locator('input.wfid').fill('FromA')
  await a.getByRole('button', { name: /^Save/ }).click()
  await expect(a.getByText(`Saved to ${WF}`)).toBeVisible()
  expect(await serverId(request)).toBe('FromA')

  await b.locator('input.wfid').fill('FromB')
  await b.getByRole('button', { name: /^Save/ }).click()
  const dialog = b.getByRole('alertdialog')
  await expect(dialog).toContainText(
    `${WF} changed since you opened it, in another tab or by an agent. Replace that version with yours?`,
  )
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  expect(await serverId(request)).toBe('FromA')
})

test('prompt editor: text typed and saved at once survives a reload', async ({
  page,
  request,
}) => {
  const saved = await request.put(`/api/prompts/${PROMPT}`, {
    data: { prompt: { text: 'first' } },
  })
  expect(saved.status()).toBe(200)
  await page.goto(`/#/shared/prompt-edit/${PROMPT}`)
  const text = page.locator('#prompt-text')
  await expect(text).toHaveValue('first')
  await text.fill('typed and saved at once')
  await page.keyboard.press('ControlOrMeta+s')
  await expect(page.getByText(`Saved to ${PROMPT}`)).toBeVisible()
  const read = await (await request.get(`/api/prompts/${PROMPT}`)).json()
  expect(read.text).toBe('typed and saved at once')
  await page.reload()
  await expect(page.locator('#prompt-text')).toHaveValue(
    'typed and saved at once',
  )
})

test('assets: one dialog for a batch with a taken name, and Keep both makes name-2', async ({
  page,
  request,
}) => {
  const png = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAfbLI3wAAAABJRU5ErkJggg==',
    'base64',
  )
  const taken = await request.post(
    '/api/uploads?filename=e2e-safety-a.png&asset_name=e2e-safety-a.png',
    { data: png },
  )
  expect(taken.ok()).toBeTruthy()
  try {
    await page.goto('/#/ws/default/assets')
    await page.locator('input[type=file]').setInputFiles([
      { name: 'e2e-safety-a.png', mimeType: 'image/png', buffer: png },
      { name: 'e2e-safety-b.png', mimeType: 'image/png', buffer: png },
    ])
    const dialog = page.getByRole('alertdialog')
    await expect(dialog).toContainText(
      '1 of 2 files already exist in the library.',
    )
    await expect(page.getByRole('alertdialog')).toHaveCount(1)
    await dialog.getByRole('button', { name: 'Keep both' }).click()
    await expect(dialog).toHaveCount(0)
    await expect
      .poll(async () => {
        const list = await (await request.get('/api/assets')).json()
        return JSON.stringify(list)
      })
      .toContain('e2e-safety-a-2.png')
    const names = JSON.stringify(
      await (await request.get('/api/assets')).json(),
    )
    expect(names).toContain('e2e-safety-b.png')
  } finally {
    for (const n of ['a', 'b', 'a-2'])
      await request.delete(`/api/assets/uploads/e2e-safety-${n}.png`)
  }
})
