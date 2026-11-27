import { expect, test } from './test'

test('a long prompt card is clamped to two lines', async ({ page }) => {
  // The real listing, with every prompt's description made far too long
  await page.route('**/api/prompts?times=true', async (route) => {
    const response = await route.fetch()
    const json = await response.json()
    for (const detail of Object.values<{ description: string }>(json.details)) {
      detail.description = 'a lighthouse in heavy weather '.repeat(60)
    }
    await route.fulfill({ response, json })
  })
  await page.goto('/#/shared/prompts')
  const text = page.locator('.card .carddesc').first()
  await expect(text).toBeVisible()
  const { height, line, full } = await text.evaluate((el) => ({
    height: el.getBoundingClientRect().height,
    line: parseFloat(getComputedStyle(el).lineHeight),
    full: el.scrollHeight,
  }))
  expect(height).toBeLessThanOrEqual(line * 2 + 1)
  expect(height).toBeGreaterThan(line)
  expect(full).toBeGreaterThan(height)
})
