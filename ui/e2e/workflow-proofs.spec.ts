import { expect, test } from './test'

// A workflow's page shows what it last made (the proofs strip) from a read
// of that workflow's folder alone, capped at six (#1701): against the real
// server, the narrowed answer reaches the strip. The strip is pictures only
// (caption="none"), so a tile is found by its link, alt and title, never by
// text. The fixture's `card-proof` workflow has one output.

test('the proofs strip shows the folder read, reached from the list', async ({
  page,
}) => {
  const reads: string[] = []
  page.on('request', (r) => {
    const url = new URL(r.url())
    if (url.pathname === '/api/gallery') reads.push(url.search)
  })
  await page.goto('/#/ws/default/workflows')
  await page.locator('a', { hasText: 'card-proof' }).first().click()
  await expect(page).toHaveURL(/#\/ws\/default\/workflows\/card-proof$/)

  const tile = page.locator('.proofs a.filetile')
  await expect(tile).toHaveCount(1)
  await expect(tile).toHaveAttribute(
    'title',
    'proof.png — open it in the gallery',
  )
  await expect(tile.locator('img')).toHaveAttribute('alt', 'proof.png')
  expect(reads).toContain('?folder=card-proof&limit=6')
})

// A workflow under a folder (templates/image-processors): its identity has a
// slash, its route has two segments, and its proof sits in a run's `final`
// subfolder (#1769)
test('the proofs strip shows a slash-named folder read', async ({ page }) => {
  const reads: string[] = []
  page.on('request', (r) => {
    const url = new URL(r.url())
    if (url.pathname === '/api/gallery') reads.push(url.search)
  })
  await page.goto('/#/ws/default/workflows')
  await page.locator('a', { hasText: 'image-processors' }).first().click()
  await expect(page).toHaveURL(
    /#\/ws\/default\/workflows\/templates\/image-processors$/,
  )

  const tile = page.locator('.proofs a.filetile')
  await expect(tile).toHaveCount(1)
  await expect(tile).toHaveAttribute(
    'title',
    'image_processors-resize.0-0.0.png — open it in the gallery',
  )
  expect(reads).toContain('?folder=templates%2Fimage-processors&limit=6')
})
