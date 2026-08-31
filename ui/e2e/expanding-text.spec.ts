import { expect, test } from '@playwright/test'

// A workflow typed through the JSON view: a long variable default and a
// prose-shaped task argument, so both surfaces of the expanding-text
// treatment render without any server-side pipeline imports
const LONG_PROMPT =
  'a 3/4-length studio portrait of a wiry man in his early 30s with short curly dark hair'
const DEFINITION = JSON.stringify({
  id: 'expand_check',
  variables: { prompt: LONG_PROMPT, steps: 25 },
  steps: [
    {
      name: 'gen',
      task: { command: 'noop', arguments: { caption_prompt: LONG_PROMPT } },
    },
  ],
})

async function loadDefinition(page: import('@playwright/test').Page) {
  await page.goto('/#/edit')
  await page.getByRole('button', { name: /JSON/ }).click()
  await expect(page.locator('.monaco-editor').first()).toBeVisible({
    timeout: 20_000,
  })
  await page.locator('.view-lines').first().click()
  await page.keyboard.press('ControlOrMeta+a')
  await page.keyboard.press('Backspace')
  await page.keyboard.insertText(DEFINITION)
  // Monaco applies on blur - focus something else before leaving the view
  await page.locator('input.wfid').click()
  await page.getByRole('button', { name: /form/ }).click()
}

test('a long variable expands to a document and collapses back', async ({
  page,
}) => {
  await loadDefinition(page)
  // long value renders as a single-line input with the expand affordance;
  // the short numeric variable gets no affordance
  const prompt = page.locator('input#wfvar-prompt')
  await expect(prompt).toHaveValue(LONG_PROMPT)
  await expect(
    page
      .locator('label[for="wfvar-steps"] + div')
      .getByRole('button', { name: 'expand to edit as a document' }),
  ).toHaveCount(0)
  await page
    .locator('label[for="wfvar-prompt"] + div')
    .getByRole('button', { name: 'expand to edit as a document' })
    .click()
  // now a document: a focused, growing textarea with the same id
  const doc = page.locator('textarea#wfvar-prompt')
  await expect(doc).toBeVisible()
  await expect(doc).toBeFocused()
  await doc.fill(LONG_PROMPT + ', warm even studio lighting')
  await page.getByRole('button', { name: 'collapse to one line' }).click()
  // the edit survives the round trip back to one line
  await expect(page.locator('input#wfvar-prompt')).toHaveValue(
    LONG_PROMPT + ', warm even studio lighting',
  )
})

test('long prose arguments collapse to a line and expand on demand', async ({
  page,
}) => {
  await loadDefinition(page)
  // loaded steps open compact - get to the editable fields
  await page.getByRole('button', { name: 'full' }).first().click()
  const row = page.locator('.args .row', { hasText: 'caption_prompt' })
  // one line with the affordance, not the old always-three-rows textarea
  await expect(row.locator('input')).toBeVisible()
  await row
    .getByRole('button', { name: 'expand to edit as a document' })
    .click()
  await expect(row.locator('textarea')).toBeVisible()
  // the expanded view offers the stored-prompt library
  await expect(row.locator('select.promptpick')).toBeVisible()
})

test('the run page shares the same field treatment', async ({ page }) => {
  // FluxDev's prompt default is a stored-prompt reference - short - but
  // guidance-style workflows aside, ZImage's prompt default is prose; use
  // the editor-saved definition instead to stay deterministic
  await page.goto('/#/workflows/ZImage')
  await expect(page.getByLabel('num_inference_steps')).toBeVisible()
  // overrides render through the shared form: same grid, same input ids
  await expect(page.locator('input#var-prompt')).toBeVisible()
})
