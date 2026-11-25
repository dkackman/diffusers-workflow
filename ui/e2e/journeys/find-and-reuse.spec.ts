import { expect, test } from '@playwright/test'

// The journey of the media phase, end to end against the real server and
// the fixture's e2e-find run: find a file in the Gallery, open it and read
// what made it, take it further (its reference, an asset, a star, a
// comparison), then meet it again from the Overview and the job that made it.
const GALLERY = '/#/ws/default/gallery'
const RUN = 'e2e-find/20260910-120000-0f1d0f1d'
const LIGHTHOUSE = `${RUN}/lighthouse.png`
const HARBOUR = `${RUN}/harbour.png`

test('find a file, read what made it, and reuse it', async ({
  page,
  context,
  request,
}) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  await page.goto(GALLERY)

  // 1. The search reads a run's prompt, which every file of the run answers
  // to; naming the file as well leaves one tile, and the count says so
  const find = page.getByPlaceholder('search names, prompts and notes…')
  await find.fill('lighthouse dusk')
  await expect(page.locator('.filetile')).toHaveCount(4)
  await find.fill('lighthouse.png dusk')
  await expect(page.locator('.filetile')).toHaveCount(1)
  await expect(page.getByText(/^1 of \d+ files$/)).toBeVisible()

  // The viewer steps through what the grid shows, so widen it back to the
  // run before opening: the arrows in step 3 need a neighbour
  await find.fill('lighthouse dusk')
  await expect(page.locator('.filetile')).toHaveCount(4)

  // 2. Opening it puts open= in the URL, fits the picture to the window and
  // shows what made it without a click
  await page.locator('.filetile', { hasText: 'lighthouse.png' }).click()
  await expect(page).toHaveURL(/open=/)
  const viewer = page.getByRole('dialog', { name: 'lighthouse.png' })
  await expect(viewer.getByRole('button', { name: 'Close' })).toBeVisible()
  const picture = viewer.locator('img.media-body')
  await expect(picture).toBeVisible()
  const window_ = page.viewportSize()!
  await expect
    .poll(async () => (await picture.boundingBox())!.height)
    .toBeLessThanOrEqual(window_.height)
  await expect(viewer.getByText('What made this')).toBeVisible()
  await expect(viewer.getByText('a lighthouse at dusk').first()).toBeVisible()
  await expect(viewer.getByText('Seed')).toBeVisible()
  await expect(viewer.getByText('4242').first()).toBeVisible()

  // 3. Zoom in and back to fit; the arrows step and return
  await viewer.getByRole('button', { name: 'Zoom in' }).click()
  await expect(
    viewer.getByRole('button', { name: 'Fit to window' }),
  ).toBeVisible()
  await viewer.getByRole('button', { name: 'Fit to window' }).click()
  await expect(
    viewer.getByRole('button', { name: /^Actual size/ }),
  ).toBeVisible()
  const position = page.locator('.viewer-bar .num')
  const first = (await position.textContent())!
  // The file may stand last in its list: then → has nowhere to go, so ← first
  const [at, of] = first.split(' of ').map(Number)
  const [there, back] =
    at < of ? ['ArrowRight', 'ArrowLeft'] : ['ArrowLeft', 'ArrowRight']
  await page.keyboard.press(there)
  await expect(position).not.toHaveText(first)
  await page.keyboard.press(back)
  await expect(position).toHaveText(first)

  // 3b. Escape closes the viewer and takes open= away; reopened, Back
  // closes it with the grid in place. Neither leaves an entry behind: one
  // more Back leaves the Gallery, and Forward brings it back for step 4
  const lighthouse = page.locator('.filetile', { hasText: 'lighthouse.png' })
  await page.keyboard.press('Escape')
  await expect(viewer).toBeHidden()
  await expect(page).not.toHaveURL(/open=/)
  await lighthouse.click()
  await expect(viewer).toBeVisible()
  await expect(page).toHaveURL(/open=/)
  await page.goBack()
  await expect(viewer).toBeHidden()
  await expect(page).not.toHaveURL(/open=/)
  await expect(page).toHaveURL(/\/gallery/)
  await expect(page.locator('.filetile')).toHaveCount(4)
  await page.goBack()
  await expect(page).not.toHaveURL(/\/gallery/)
  await page.goForward()
  await expect(page.locator('.filetile')).toHaveCount(4)
  await lighthouse.click()
  await expect(viewer).toBeVisible()

  // 4. The reference chip copies the text a workflow argument takes
  await viewer
    .getByRole('button', {
      name: 'Copy the reference a workflow argument takes',
    })
    .click()
  await expect
    .poll(() => page.evaluate(() => navigator.clipboard.readText()))
    .toBe(`output:${LIGHTHOUSE}`)

  // 5. Save as asset: its tile is a thumbnail, and it opens with Saved from
  // The asset goes again whatever happens: a repeat saves the same name
  try {
    await viewer.getByRole('button', { name: 'Save as asset' }).click()
    const naming = page.getByRole('alertdialog', { name: 'Save as asset' })
    await naming.getByRole('textbox').fill('lighthouse-ref')
    await naming.getByRole('button', { name: 'Save' }).click()
    await expect(page.getByText('Saved as asset lighthouse-ref')).toBeVisible()
    await page.goto('/#/ws/default/assets')
    const tile = page.locator('.filetile', { hasText: 'lighthouse-ref' })
    await expect(tile).toBeVisible()
    await expect(tile.locator('video')).toHaveCount(0)
    const source = await tile.locator('img').first().getAttribute('src')
    expect(source).toContain('/thumbnail')
    expect(source).not.toMatch(/\/assets\/lighthouse-ref/)
    await tile.click()
    await expect(page.getByText('Saved from').first()).toBeVisible()
  } finally {
    await request.delete('/api/assets/lighthouse-ref.png')
  }

  // 6. Star two files, then Undo: neither is starred
  await page.goto(GALLERY)
  await page.getByLabel(`select ${LIGHTHOUSE}`).check()
  await page.getByLabel(`select ${HARBOUR}`).check()
  await page.getByRole('button', { name: 'Star', exact: true }).click()
  await expect(page.getByText('Starred 2 files')).toBeVisible()
  await page.getByRole('button', { name: 'Undo' }).click()
  await expect
    .poll(async () => {
      const read = await request.get('/api/gallery')
      const files: { name: string; starred?: boolean }[] = (await read.json())
        .files
      return files
        .filter((f) => [LIGHTHOUSE, HARBOUR].includes(f.name))
        .map((f) => !!f.starred)
    })
    .toEqual([false, false])

  // 7. Compare the two, remove one: one name left, and the empty copy
  await page.getByRole('button', { name: /^compare the selected/ }).click()
  await expect(page).toHaveURL(/compare/)
  await expect(
    page.getByRole('button', { name: 'Remove from comparison' }),
  ).toHaveCount(2)
  await page
    .getByRole('button', { name: 'Remove from comparison' })
    .first()
    .click()
  // The names are the path after /compare/
  await expect
    .poll(() => {
      const rest = page.url().split('/compare/')[1] ?? ''
      return rest.split('/').filter(Boolean).map(decodeURIComponent)
    })
    .toHaveLength(1)
  await expect(page.getByText('Nothing to compare yet.')).toBeVisible()

  // 8. The Overview's recent tile opens the file in the Gallery's viewer.
  // The row holds the newest files, and what ran before this decides which
  // those are: the real listing is served with the lighthouse first
  await page.route(
    (url) => url.pathname === '/api/gallery',
    async (route) => {
      const response = await route.fetch()
      const body: { files: { name: string }[] } = await response.json()
      body.files.sort(
        (a, b) => Number(b.name === LIGHTHOUSE) - Number(a.name === LIGHTHOUSE),
      )
      await route.fulfill({ response, json: body })
    },
  )
  await page.goto('/#/ws/default/overview')
  await page.getByRole('link', { name: 'lighthouse.png', exact: true }).click()
  await expect(page).toHaveURL(/gallery/)
  await expect(page).toHaveURL(/open=/)
  await expect(
    page.getByRole('dialog', { name: 'lighthouse.png' }),
  ).toBeVisible()

  // 9. The job that made it: Results come before the workflow, and its tile
  // opens the same viewer
  await page.goto('/#/ws/default/jobs/e2efind00001')
  const results = page.getByRole('heading', { name: 'Results' })
  const flow = page.getByRole('heading', { name: 'Workflow' })
  await expect(results).toBeVisible()
  const [resultsAt, flowAt] = [
    await results.boundingBox(),
    await flow.boundingBox(),
  ]
  expect(resultsAt!.y).toBeLessThan(flowAt!.y)
  await page.locator('.filetile', { hasText: 'lighthouse' }).first().click()
  await expect(
    page.getByRole('dialog', { name: 'lighthouse.png' }),
  ).toBeVisible()

  // 10. The clip's length and the note's first line, on their tiles
  await page.goto(`${GALLERY}?q=e2e-find`)
  await expect(page.locator('.filetile', { hasText: 'hum.wav' })).toContainText(
    '2.0s',
  )
  await expect(
    page.locator('.filetile', { hasText: 'notes.txt' }),
  ).toContainText("Keeper's log, first night")
})
