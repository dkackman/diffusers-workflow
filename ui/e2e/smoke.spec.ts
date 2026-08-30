import { expect, test } from '@playwright/test'

test('workflow browser lists, describes and filters', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Workflows' })).toBeVisible()
  // cards carry descriptions from the sweep
  await expect(
    page.getByText('Text-to-image with Z-Image Turbo').first(),
  ).toBeVisible()
  // the filter searches descriptions, not just names
  await page.getByPlaceholder('filter…').fill('inpaint')
  await expect(page.getByRole('link', { name: /FluxFill/ })).toBeVisible()
  await expect(page.getByRole('link', { name: /^ZImage / })).toHaveCount(0)
})

test('workflow page shows JSON and a run form', async ({ page }) => {
  await page.goto('/#/workflows/ZImage')
  await expect(page.getByRole('heading', { name: 'ZImage' })).toBeVisible()
  // JSON is shown by default via Monaco
  await expect(page.locator('.monaco-editor').first()).toBeVisible({
    timeout: 20_000,
  })
  // the variables form renders with defaults as placeholders
  await expect(page.getByLabel('num_inference_steps')).toBeVisible()
  await expect(page.getByRole('button', { name: /Run/ })).toBeEnabled()
})

test('editor opens a workflow with introspected arguments', async ({
  page,
}) => {
  test.setTimeout(120_000) // first describe imports the pipeline class server-side
  await page.goto('/#/edit/flux/FluxDev')
  await expect(page.locator('#ct-0')).toHaveValue('FluxPipeline')
  // the arguments editor discovered real __call__ parameters - the
  // add-argument select renders once the description arrives
  await expect(
    page.locator('select').filter({ hasText: 'add argument…' }).first(),
  ).toBeVisible({ timeout: 90_000 })
  // reference values render as references (accent+mono styling)
  const promptField = page.locator('input.ref').first()
  await expect(promptField).toHaveValue('variable:prompt')
})

test('split view shows editable JSON beside the form', async ({ page }) => {
  await page.goto('/#/edit/ZImage')
  await page.getByRole('button', { name: /split/ }).click()
  await expect(page.locator('.jsoncol')).toBeVisible()
  // the live definition renders in a real editor
  await expect(page.locator('.jsoncol .monaco-editor')).toBeVisible({
    timeout: 20_000,
  })
  // the pane is editable: replace the JSON and the form follows on blur
  await page.locator('.jsoncol .view-lines').click()
  await page.keyboard.press('ControlOrMeta+a')
  await page.keyboard.press('Backspace')
  await page.keyboard.insertText(
    '{"id": "typed_in_json", "variables": {}, "steps": []}',
  )
  await expect(page.locator('.jsoncol .view-lines')).toContainText(
    'typed_in_json',
  )
  await page.locator('h2', { hasText: 'Variables' }).click()
  await expect(page.locator('input.wfid')).toHaveValue('typed_in_json')
  // views are mutually exclusive - form closes the pane
  await page.getByRole('button', { name: /form/ }).click()
  await expect(page.locator('.jsoncol')).toHaveCount(0)
})

test('schema page renders a browsable tree from the live schema', async ({
  page,
}) => {
  await page.goto('/#/schema')
  await expect(
    page.getByRole('heading', { name: 'Workflow Schema' }),
  ).toBeVisible()
  // the document root is open and shows the top-level properties
  await expect(page.getByText('steps', { exact: true })).toBeVisible()
  // expanding a definition reveals its properties
  await page.locator('#def-step').getByRole('button').first().click()
  await expect(
    page.locator('#def-step').getByText('pipeline', { exact: true }).first(),
  ).toBeVisible()
  // the definitions filter narrows the list
  await page.getByPlaceholder('filter…').fill('lora')
  await expect(page.locator('#def-lora')).toBeVisible()
  await expect(page.locator('#def-step')).toHaveCount(0)
})

test('editor validates, saves into a folder, and deletes', async ({ page }) => {
  test.setTimeout(60_000)
  await page.goto('/#/edit/ZImage')
  await page.getByRole('button', { name: 'Validate' }).click()
  await expect(page.getByText('schema-valid')).toBeVisible({ timeout: 30_000 })

  // save under a scratch name, then clean up through the UI's own delete
  await page.getByPlaceholder('MyWorkflow').fill('E2EScratch')
  await page.getByRole('button', { name: 'Save' }).click()
  await expect(page.getByText(/Saved to/)).toBeVisible({ timeout: 30_000 })

  await page.goto('/#/workflows/E2EScratch')
  page.once('dialog', (dialog) => dialog.accept())
  await page.getByRole('button', { name: /delete this workflow/ }).click()
  await expect(page.getByRole('heading', { name: 'Workflows' })).toBeVisible()
  await expect(page.getByRole('link', { name: 'E2EScratch' })).toHaveCount(0)
})

test('jobs and gallery pages render', async ({ page }) => {
  await page.goto('/#/jobs')
  await expect(page.getByRole('heading', { name: 'Jobs' })).toBeVisible()
  await page.goto('/#/gallery')
  await expect(page.getByRole('heading', { name: 'Gallery' })).toBeVisible()
})

test('validation flags a signature typo through the real server', async ({
  page,
}) => {
  test.setTimeout(90_000)
  // through the same origin the UI uses - schema-valid, signature-wrong
  const broken = {
    id: 'typo_check',
    steps: [
      {
        name: 'gen',
        pipeline: {
          configuration: { component_type: 'ZImagePipeline' },
          from_pretrained_arguments: { model_name: 'x' },
          arguments: { prompt: 'p', guidance_scael: 3 },
        },
      },
    ],
  }
  const response = await page.request.post('/api/validate', {
    data: { workflow: broken },
    timeout: 60_000,
  })
  const result = await response.json()
  expect(result.valid).toBe(true)
  expect(result.warnings.join(' ')).toContain('guidance_scael')
})

test('models page inventories the hub cache', async ({ page }) => {
  // Read-only: this runs against the developer's real hub cache
  await page.goto('/#/models')
  await expect(page.locator('h1')).toHaveText('Models')
  await expect(page.locator('.head .muted')).toContainText('cached', {
    timeout: 30_000,
  })
  // The download form is present but never submitted here
  await expect(page.getByPlaceholder(/download a model/)).toBeVisible()
})

test('task steps get introspection-driven forms', async ({ page }) => {
  await page.goto('/#/edit/tasks/ImageToText')
  // the command's discovered schema renders labeled fields
  await expect(page.locator('label', { hasText: 'image' }).first()).toBeVisible(
    { timeout: 30_000 },
  )
  // image_to_text takes **kwargs, so the free-form add is offered too
  await expect(
    page.getByPlaceholder('add custom argument…').first(),
  ).toBeVisible()
})
