import { expect, test, type Page } from '../test'

// The journey of the editor phase, end to end against the real server: chain
// a second step onto an example's result, rename the step it reads and see
// the reference follow, be told no to a bad name, break a reference and be
// taken to it, add a variable, and fill a file argument by choosing an
// asset. Nothing is saved: the draft is the browser's. The asset is one the
// spec keeps from the fixture's e2e-find run and removes again.
const ASSET = 'e2e-chain-harbour'
const HARBOUR = 'e2e-find/20260910-120000-0f1d0f1d/harbour.png'

const SECOND = 'StableDiffusionImg2ImgPipeline'

test.beforeEach(async ({ request }) => {
  // The class is described once before the page asks: its first import
  // racing the live check's on a cold server is the import-lock flake the
  // suite runs on one worker for (playwright.config.ts)
  const described = await request.get(`/api/pipelines/${SECOND}`, {
    timeout: 60_000,
  })
  expect(described.status()).toBe(200)
  const kept = await request.post('/api/assets/keep', {
    data: { name: HARBOUR, asset_name: ASSET, overwrite: true },
  })
  expect(kept.status()).toBe(201)
})

/** The detail level each step is shown at, in order */
async function detailLevels(page: Page) {
  const steps = page.locator('.panel.step')
  const levels: string[] = []
  for (let at = 0; at < (await steps.count()); at++) {
    const pressed = steps
      .nth(at)
      .getByRole('group', { name: 'step detail level' })
      .locator('button[aria-pressed="true"]')
    levels.push((await pressed.textContent())!.trim())
  }
  return levels
}

test('chain a step, rename what it reads, and fill it from the assets', async ({
  page,
  request,
}) => {
  try {
    // 1. A new workflow from the text-to-image example (the fixture's
    // examples are its workflows, so Duplicate starts the copy), and a
    // pipeline step after it whose image argument offers the first step's
    // result first
    await page.goto('/#/ws/default/workflows/templates/text-to-image')
    await page.getByRole('button', { name: 'Duplicate' }).click()
    await expect(page).toHaveURL(/\/edit$/)
    const steps = page.locator('.panel.step')
    await expect(steps).toHaveCount(1)
    await expect(steps.first().locator('input.name')).toHaveValue('main')
    await page.getByRole('button', { name: 'pipeline step' }).click()
    await expect(steps).toHaveCount(2)
    const second = steps.nth(1)
    await second.getByLabel('pipeline', { exact: true }).fill(SECOND)
    await second
      .getByLabel('model', { exact: true })
      .fill('stable-diffusion-v1-5/stable-diffusion-v1-5')
    await second.getByRole('button', { name: /available arguments/ }).click({
      timeout: 20_000,
    })
    await second.getByRole('button', { name: 'add image', exact: true }).click()
    const image = second.getByLabel('image', { exact: true })
    await expect(image).toHaveValue('')
    await image.focus()
    await page.keyboard.press('ArrowDown')
    const rows = page.locator('.ui-suggest').getByRole('option')
    await expect(rows.first()).toContainText('previous_result:main')
    await page.keyboard.press('Enter')
    await expect(image).toHaveValue('previous_result:main')
    await expect(image).toHaveClass(/referencefield/)
    const levels = await detailLevels(page)

    // 2. Renaming the step carries the reference to it, and each step keeps
    // the detail level it had
    const name = steps.first().locator('input.name')
    await name.fill('still')
    await name.press('Enter')
    await expect(
      page.getByText('Renamed step main to still, and 1 reference to it'),
    ).toBeVisible()
    await expect(image).toHaveValue('previous_result:still')
    expect(await detailLevels(page)).toEqual(levels)

    // 3. A name the engine would refuse is said, and the step keeps its own
    await name.fill('bad name')
    await name.press('Enter')
    await expect(
      page.getByText("A step name can't be empty or hold a space, . or @."),
    ).toBeVisible()
    await expect(name).toHaveValue('still')

    // 4. A reference to a variable that is gone is found while typing, and
    // Go to takes the person to the place it names
    const prompt = page.locator('.vars label', { hasText: /^prompt$/ })
    await expect(prompt).toBeVisible()
    await page
      .locator('.vars')
      .getByRole('button', { name: 'remove this variable' })
      .first()
      .click()
    await expect(page.locator('#wfvar-prompt')).toHaveCount(0)
    const finding = page
      .locator('.error')
      .filter({ hasText: "'variable:prompt' names no declared variable" })
    const goTo = finding.getByRole('button', { name: 'Go to step still' })
    await expect(goTo).toBeVisible({ timeout: 2_000 })
    await goTo.click()
    await expect(page.locator('#step-still input.name')).toBeFocused()

    // 5. A variable added on Enter takes its type from what is typed
    const fresh = page.getByPlaceholder('new variable name…')
    await fresh.fill('steps')
    await fresh.press('Enter')
    const value = page.locator('#wfvar-steps')
    await value.fill('30')
    await value.blur()
    await page.getByRole('button', { name: 'JSON', exact: true }).click()
    await expect(page.locator('.monaco-editor').first()).toBeVisible({
      timeout: 20_000,
    })
    await expect(page.locator('.view-lines').first()).toContainText(
      '"steps": 30',
    )
    await page.getByRole('button', { name: 'form', exact: true }).click()

    // 6. The second step's image, chosen from the assets: an asset: reference,
    // shown as a tile
    const field = page
      .locator('.panel.step')
      .nth(1)
      .locator('.reference')
      .filter({ has: page.getByLabel('image', { exact: true }) })
    await field.getByRole('button', { name: 'Choose…' }).click()
    const picker = page.getByRole('dialog', { name: 'choose a file' })
    await picker.getByRole('button', { name: 'Assets', exact: true }).click()
    await picker.getByRole('button', { name: `choose ${ASSET}.png` }).click()
    await expect(picker).toBeHidden()
    await expect(field.getByRole('combobox')).toHaveValue(`asset:${ASSET}.png`)
    await expect(field.locator('.preview .filetile')).toBeVisible()
  } finally {
    await request.delete(`/api/assets/${ASSET}.png`)
  }
})
