import { expect, test } from '@playwright/test'

// The journey of the run-page phase, end to end against the real server:
// find the workflow by what it does, read whether it will work, fix a value
// the check refuses, leave and come back to what was typed, run it, and
// fill a file field by choosing a picture an earlier job made. The job
// itself is the fixture's e2efind00001: the POST is answered with it, so
// nothing runs on a GPU.
const RUN = 'e2e-find/20260910-120000-0f1d0f1d'
const LIGHTHOUSE = `${RUN}/lighthouse.png`

test('find a workflow, check it, run it and fill a file field', async ({
  page,
  request,
}) => {
  // 1. Word search puts the best match first, and the count says how many
  await page.goto('/#/ws/default/workflows')
  await page.getByPlaceholder('search workflows…').fill('text to image')
  await expect(page.getByText(/^\d+ of \d+ workflows$/)).toBeVisible()
  const cards = page.locator('a.card')
  await expect(cards.first().locator('.cardname')).toHaveText('text-to-image')
  // Three templates share the name, so searching puts each one's folder on
  // its card; the baseline is the one in templates/ itself
  const baseline = page.locator('a.card[href$="/templates/text-to-image"]')
  await expect(baseline.locator('.cardfolder')).toHaveText('templates')

  // 2. Open it: the check says it will work, the rarely changed fields are
  // folded, and the seed with no default says what it does instead
  await baseline.click()
  await expect(page).toHaveURL(/\/workflows\/templates\/text-to-image$/)
  const check = page.getByRole('region', { name: 'run check' })
  await expect(check.getByText('Ready to run.', { exact: true })).toBeVisible({
    timeout: 20_000,
  })
  const advanced = page.getByRole('button', { name: /^Advanced \(\d+\)$/ })
  await expect(advanced).toHaveAttribute('aria-expanded', 'false')
  await advanced.click()
  await expect(page.locator('#var-seed')).toHaveAttribute(
    'placeholder',
    'random each job',
  )

  // 3. A cost driver is an Essential: a value the check refuses turns Run
  // into Fix, which goes to the field; a good value brings Run back
  const count = page.locator('#var-num_images_per_prompt')
  await count.fill('abc')
  await count.blur()
  const fix = page.getByRole('button', { name: 'Fix 1 issue' })
  await expect(fix).toBeVisible({ timeout: 20_000 })
  await fix.click()
  await expect(count).toBeFocused()
  await count.fill('2')
  await count.blur()
  const run = page.getByRole('button', { name: 'Run', exact: true })
  await expect(run).toBeVisible({ timeout: 20_000 })

  // 4. What was typed survives leaving the page and coming back
  const prompt = page.locator('#var-prompt')
  await prompt.fill('a lighthouse at dusk')
  await prompt.blur()
  const here = page.url()
  await page.getByRole('link', { name: 'Gallery' }).first().click()
  await expect(page).toHaveURL(/\/gallery/)
  await expect(page.locator('#var-prompt')).toHaveCount(0)
  await page.goto(here)
  await expect(page.locator('#var-prompt')).toHaveValue('a lighthouse at dusk')
  await expect(
    page.getByText('Restored what you last typed here.'),
  ).toBeVisible()

  // 5. Run sends what the form holds, typed: the count as a number. The
  // job page that opens is the fixture's job, with its results
  const job = await (await request.get('/api/jobs/e2efind00001')).json()
  let sent: { arguments?: Record<string, unknown> } | null = null
  await page.route('**/api/jobs', async (route) => {
    if (route.request().method() !== 'POST') return route.continue()
    sent = route.request().postDataJSON()
    await route.fulfill({ status: 201, json: job })
  })
  await expect(run).toBeEnabled({ timeout: 20_000 })
  await run.click()
  await expect(page).toHaveURL(/\/jobs\/e2efind00001/)
  expect(sent!.arguments).toMatchObject({
    num_images_per_prompt: 2,
    prompt: 'a lighthouse at dusk',
  })
  await expect(page.getByRole('heading', { name: 'Results' })).toBeVisible()
  await expect(
    page.locator('.filetile', { hasText: 'lighthouse.png' }).first(),
  ).toBeVisible()

  // 6. A file field offers Upload… and Choose…; Choose shows only what
  // the field takes, and a picture an earlier job made lands as its
  // output: reference, shown as a tile
  await page.goto('/#/ws/default/workflows/templates/inpaint')
  const image = page.locator('#var-image')
  await expect(image).toBeVisible({ timeout: 20_000 })
  const field = page.locator('.reference').filter({ has: image })
  await expect(field.getByRole('button', { name: 'Upload…' })).toBeVisible()
  await field.getByRole('button', { name: 'Choose…' }).click()
  const picker = page.getByRole('dialog', { name: 'choose a file' })
  await expect(
    picker.getByRole('heading', { name: 'Choose an image' }),
  ).toBeVisible()
  await picker.getByRole('button', { name: 'Earlier jobs' }).click()
  const pick = picker.getByRole('button', { name: `choose ${LIGHTHOUSE}` })
  await expect(pick).toBeVisible()
  // images only: the clip and the note the same run wrote are not offered
  await expect(
    picker.getByRole('button', { name: `choose ${RUN}/hum.wav` }),
  ).toHaveCount(0)
  await expect(
    picker.getByRole('button', { name: `choose ${RUN}/notes.txt` }),
  ).toHaveCount(0)
  await pick.click()
  await expect(picker).toBeHidden()
  await expect(image).toHaveValue(`output:${LIGHTHOUSE}`)
  await expect(field.locator('.preview .filetile')).toBeVisible()
})
