// ui/e2e/step-model.spec.ts
import { expect, test } from '@playwright/test'

// A two-step task workflow typed through the JSON view - task steps need
// no server-side pipeline imports, so these tests stay fast
const TWO_STEP = JSON.stringify({
  id: 'flow_check',
  variables: {},
  steps: [
    { name: 'first', task: { command: 'noop', arguments: { x: 1 } } },
    {
      name: 'second',
      task: { command: 'noop', arguments: { y: 'previous_result:first' } },
    },
  ],
})

async function loadTwoStep(page: import('@playwright/test').Page) {
  await page.goto('/#/edit')
  await page.getByRole('button', { name: /JSON/ }).click()
  await expect(page.locator('.monaco-editor').first()).toBeVisible({
    timeout: 20_000,
  })
  await page.locator('.view-lines').first().click()
  await page.keyboard.press('ControlOrMeta+a')
  await page.keyboard.press('Backspace')
  await page.keyboard.insertText(TWO_STEP)
  // changes apply on blur - move focus off the editor before switching views,
  // otherwise the JsonEditor unmounts before its blur handler applies the edit
  await page.locator('input.wfid').click()
  await page.getByRole('button', { name: /form/ }).click()
}

test('steps carry ordinals and producer/consumer chips', async ({ page }) => {
  await loadTwoStep(page)
  // the rail numbers the sequence
  await expect(page.locator('.ordinal').nth(0)).toHaveText('1')
  await expect(page.locator('.ordinal').nth(1)).toHaveText('2')
  // fan edges render on both ends
  await expect(
    page.locator('.flowchip.out', { hasText: 'second' }),
  ).toBeVisible()
  await expect(page.locator('.flowchip.in', { hasText: 'first' })).toBeVisible()
  // hovering the consumer's input chip lights the producer's row
  await page.locator('.flowchip.in', { hasText: 'first' }).hover()
  await expect(page.locator('.steprow.flowlit')).toHaveCount(1)
})

test('step density cycles collapsed / compact / full and persists intent', async ({
  page,
}) => {
  await loadTwoStep(page)
  const firstStep = page.locator('.panel.step').first()
  // compact digest shows what is set, as text
  await firstStep.getByRole('button', { name: 'compact' }).click()
  await expect(firstStep.locator('.digestline')).toContainText('x = 1')
  // clicking a digest line jumps to full view
  await firstStep.locator('.digestline').first().click()
  await expect(firstStep.getByLabel('x')).toBeVisible()
  // collapsing leaves a one-line summary, not nothing
  await firstStep.getByRole('button', { name: /collapse this step/ }).click()
  await expect(firstStep.locator('.summary')).toContainText('task: noop')
})

test('collapse all / expand all sweep every step', async ({ page }) => {
  await loadTwoStep(page)
  await page.getByRole('button', { name: 'collapse all' }).click()
  await expect(page.locator('.summary')).toHaveCount(2)
  await page.getByRole('button', { name: 'expand all' }).click()
  await expect(page.locator('.summary')).toHaveCount(0)
})

test('reordering that breaks a reference warns on the step itself', async ({
  page,
}) => {
  await loadTwoStep(page)
  await expect(page.locator('.stepwarn')).toHaveCount(0)
  // move the consumer above its producer
  await page
    .locator('.panel.step')
    .nth(1)
    .getByRole('button', { name: 'move up' })
    .click()
  await expect(page.locator('.stepwarn')).toContainText(
    'no earlier step has that name',
  )
})
