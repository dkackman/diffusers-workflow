import { expect, test, type Locator, type Page } from './test'

// The viewer in real full screen (#1658): the browser paints nothing outside
// the full-screen element, so what opens over the viewer has to be inside it
// and on top there.

const NAME = 'clip/run-1/final/still.png'

async function mockGallery(page: Page) {
  await page.route('**/api/gallery?**', (route) =>
    route.fulfill({
      json: {
        files: [
          {
            name: NAME,
            folder: 'clip',
            subfolder: 'final',
            run_id: 'run-1',
            version: 1,
            url: `/outputs/${NAME}`,
            kind: 'image',
            size: 2048,
            mtime: 1_700_000_000,
            label: 'still.png',
            role: 'final',
          },
        ],
        folders: ['clip'],
        subfolders: ['final'],
        limit: 200,
        offset: 0,
        total: 1,
        workspace: 'default',
      },
    }),
  )
  // A 1x1 png, so the file draws and is not called missing
  await page.route('**/outputs/**', (route) =>
    route.fulfill({
      contentType: 'image/png',
      body: Buffer.from(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
        'base64',
      ),
    }),
  )
  await page.route('**/api/gallery/**/metadata**', (route) =>
    route.fulfill({
      json: {
        name: NAME,
        source: 'output',
        run_id: 'run-1',
        version: 1,
        role: 'final',
        job: null,
        metadata: null,
        media: null,
        inputs: [],
        findings: [],
      },
    }),
  )
}

/** Whether the element is what the browser draws at its own centre */
const painted = (target: Locator) =>
  target.evaluate((el) => {
    const box = el.getBoundingClientRect()
    const hit = document.elementFromPoint(
      box.x + box.width / 2,
      box.y + box.height / 2,
    )
    return (
      !!document.fullscreenElement?.contains(el) && !!hit && el.contains(hit)
    )
  })

test('the viewer in full screen shows what opens over it', async ({ page }) => {
  await mockGallery(page)
  await page.goto(`/#/ws/default/gallery?open=${encodeURIComponent(NAME)}`)
  const viewer = page.getByRole('dialog', { name: 'still.png' })
  await viewer.getByRole('button', { name: 'Full screen' }).click()
  await expect(
    viewer.getByRole('button', { name: 'Exit full screen' }),
  ).toBeVisible()

  await viewer.getByRole('button', { name: 'Use as input…' }).click()
  const use = page.getByRole('dialog', { name: 'Use still.png as input' })
  await expect.poll(() => painted(use)).toBe(true)
  await use.getByRole('button', { name: 'Cancel' }).click()
  await expect(use).toHaveCount(0)

  await viewer.getByRole('button', { name: 'Delete' }).click()
  const ask = page.getByRole('alertdialog')
  await expect.poll(() => painted(ask)).toBe(true)
  await ask.getByRole('button', { name: 'Cancel' }).click()
  await expect(ask).toHaveCount(0)
})

test('an overlay opened in full screen still closes after leaving it', async ({
  page,
}) => {
  await mockGallery(page)
  await page.goto(`/#/ws/default/gallery?open=${encodeURIComponent(NAME)}`)
  const viewer = page.getByRole('dialog', { name: 'still.png' })
  await viewer.getByRole('button', { name: 'Full screen' }).click()
  await viewer.getByRole('button', { name: 'Delete' }).click()
  const ask = page.getByRole('alertdialog')
  await expect(ask).toBeVisible()
  await page.evaluate(() => document.exitFullscreen())
  await ask.getByRole('button', { name: 'Cancel' }).click()
  await expect(ask).toHaveCount(0)
})
