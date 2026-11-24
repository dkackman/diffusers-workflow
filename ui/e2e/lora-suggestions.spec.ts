// ui/e2e/lora-suggestions.spec.ts
import { expect, test, type Page, type Route } from '@playwright/test'

// A pipeline step on a made-up base; the fixture server's LoRA library
// (ui/e2e/fixtures/loras) holds a trial and a rejected entry for it
const WORKFLOW = JSON.stringify({
  id: 'lora_suggest',
  variables: {},
  steps: [
    {
      name: 'gen',
      pipeline: {
        configuration: { component_type: 'ZImagePipeline' },
        from_pretrained_arguments: { model_name: 'e2e-org/e2e-base' },
        arguments: { prompt: 'p' },
      },
    },
  ],
})

/** Load WORKFLOW in the form editor and open its step's LoRAs section */
async function openLoras(page: Page) {
  await page.goto('/#/edit')
  await page.getByRole('button', { name: /JSON/ }).click()
  await expect(page.locator('.monaco-editor').first()).toBeVisible({
    timeout: 20_000,
  })
  await page.locator('.view-lines').first().click()
  await page.keyboard.press('ControlOrMeta+a')
  await page.keyboard.press('Backspace')
  await page.keyboard.insertText(WORKFLOW)
  await page.locator('input.wfid').click()
  await page.getByRole('button', { name: /form/ }).click()

  const step = page.locator('.panel.step').first()
  await step.getByRole('button', { name: 'full' }).click()
  await step.locator('summary', { hasText: 'LoRAs' }).click()
  return step
}

test('catalog LoRA suggestions hide rejected entries and add a row', async ({
  page,
}) => {
  const requested: string[] = []
  page.on('request', (request) => requested.push(request.url()))

  const step = await openLoras(page)
  const catalog = step.getByRole('region', { name: 'From the catalog' })
  await expect(catalog.locator('li.suggestion')).toHaveCount(1)
  await expect(catalog).not.toContainText('broken')

  // The entry's ink follows a theme switch at once, in both themes: it must
  // not lag behind the panel's colour transition (#1480, C-F1120)
  const theme = page.getByRole('button', { name: /^theme:/i })
  for (let i = 0; i < 3; i++) {
    await theme.click()
    const [ink, name, description] = await page.evaluate(() => {
      const probe = document.createElement('span')
      probe.style.color = 'var(--ink)'
      document.body.append(probe)
      const ink = getComputedStyle(probe).color
      probe.remove()
      const color = (selector: string) =>
        getComputedStyle(document.querySelector(selector)!).color
      return [
        ink,
        color('li.suggestion .name'),
        color('li.suggestion .description'),
      ]
    })
    expect(name).toBe(ink)
    expect(description).toBe(ink)
  }

  const add = catalog.getByRole('button', { name: 'add e2e/style' })
  await add.click()

  const row = step.locator('.loras .row')
  await expect(row.getByLabel('Model')).toHaveValue('e2e-org/e2e-style-lora')
  await expect(row.getByLabel('Weight file')).toHaveValue(
    'e2e_style.safetensors',
  )
  await expect(row.getByLabel('Scale')).toHaveValue('0.8')
  await expect(catalog.locator('li.suggestion button')).toHaveText('added')

  expect(
    requested.filter((url) => url.includes('/api/loras/recommend')),
  ).toEqual([])
  expect(requested.filter((url) => url.includes('huggingface.co'))).toEqual([])
})

// A failed catalog fetch shows one muted line and leaves the editor usable,
// for an HTTP error and for a request that never gets a response (C-F1119)
for (const [label, fail] of [
  [
    'a 500',
    (route: Route) =>
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'boom' }),
      }),
  ],
  ['a network error', (route: Route) => route.abort()],
] as const) {
  test(`a catalog fetch that fails with ${label} shows the muted line`, async ({
    page,
  }) => {
    let calls = 0
    await page.route('**/api/loras?**', (route) => {
      calls++
      return fail(route)
    })
    const step = await openLoras(page)
    const catalog = step.getByRole('region', { name: 'From the catalog' })
    await expect(catalog).toContainText(
      'Catalog suggestions are unavailable right now.',
    )
    expect(calls).toBeGreaterThan(0)
    await expect(catalog.locator('li.suggestion')).toHaveCount(0)

    // The editor stays usable: a row can still be added by hand
    await step.getByRole('button', { name: 'add LoRA' }).click()
    await expect(step.locator('.loras .row')).toHaveCount(1)
    await expect(catalog).toContainText(
      'Catalog suggestions are unavailable right now.',
    )
  })
}

// A request that never gets an answer must not leave the panel on a bare
// heading: it reads "Loading" first, then gives up with the failure line
test('a catalog fetch that never settles ends on the muted line', async ({
  page,
}) => {
  await page.route('**/api/loras?**', () => {
    /* never answered */
  })
  const step = await openLoras(page)
  const catalog = step.getByRole('region', { name: 'From the catalog' })
  await expect(catalog).toContainText('Loading catalog suggestions…')
  await expect(catalog).toContainText(
    'Catalog suggestions are unavailable right now.',
    { timeout: 15_000 },
  )
})
