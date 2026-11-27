import { expect, test } from './test'

// The fixture seeds two jobs a dead server's restart cancelled before they
// started, sharing its instance id (#1473). The page offers them back as one
// batch; the click previews what a rerun would queue, and cancelling queues
// nothing.
test('a restart banner previews the batch it offers to rerun', async ({
  page,
}) => {
  await page.goto('/#/ws/default/jobs')
  const banner = page.locator('.restart')
  await expect(banner).toHaveCount(1)
  await expect(banner).toContainText(
    '2 jobs were cancelled by a server restart',
  )
  // each tombstone's row carries the reason under its name
  await expect(
    page.locator('.reason', { hasText: 'server restart:' }),
  ).toHaveCount(2)

  // other journeys may have queued reruns of their own: count, don't assume none
  const reruns = page.locator('[title^="a rerun of"]')
  const before = await reruns.count()
  await banner.getByRole('button', { name: 'Rerun…' }).click()
  const dialog = page.getByRole('alertdialog')
  await expect(dialog).toBeVisible()
  // both members are listed - to queue, or skipped with a reason
  await expect(dialog).toContainText('e2erestart01')
  await expect(dialog).toContainText('e2erestart02')
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  await expect(dialog).toHaveCount(0)
  // nothing was queued: the banner still stands and no rerun row appeared
  await expect(banner).toHaveCount(1)
  await expect(reruns).toHaveCount(before)
})
