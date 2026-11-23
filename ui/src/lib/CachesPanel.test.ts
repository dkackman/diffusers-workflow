import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import CachesPanel from './CachesPanel.svelte'
import { feed } from './test/feedMock'
import { workspace } from './workspaceState.svelte'

const cache = (name: string) => ({
  name,
  root: `/cache/${name}`,
  bytes: 1024,
  files: 1,
  last_used: null,
  enabled: true,
  clearable: true,
  entries: null,
  cost_of_clearing: 'It is read again.',
  disk_free: null,
  disk_total: null,
})

const listing = vi.hoisted(() => ({ caches: [] as unknown[] }))
const listCaches = vi.hoisted(() =>
  vi.fn(() => Promise.resolve({ caches: listing.caches })),
)
const clearCache = vi.hoisted(() => vi.fn())
vi.mock('./api', () => ({
  api: { listCaches: () => listCaches(), clearCache },
}))
vi.mock('./confirm.svelte', () => ({ confirmDialog: vi.fn() }))
vi.mock('./changes.svelte', async () => ({
  changes: (await import('./test/feedMock')).feed.changes,
}))

afterEach(() => {
  cleanup()
  listCaches.mockClear()
})

it('CachesPanel reloads on a models move', async () => {
  listing.caches = [cache('triton')]
  render(CachesPanel)
  await screen.findByText('triton')
  expect(feed.watched(workspace.current)).toEqual(['models'])
  listing.caches = [cache('triton'), cache('quantized')]
  feed.move(workspace.current, 'models')
  await screen.findByText('quantized')
  expect(screen.getByText('triton')).toBeTruthy()
  expect(listCaches).toHaveBeenCalledTimes(2)
})

it("a refused clear shows the server's reason in one line, the detail folded", async () => {
  const { ApiError } = await import('./apiErrors')
  const { confirmDialog } = await import('./confirm.svelte')
  listing.caches = [cache('triton')]
  clearCache.mockRejectedValueOnce(
    new ApiError('A job is running.\nlong engine text', 409),
  )
  vi.mocked(confirmDialog).mockResolvedValue(true)
  const { container } = render(CachesPanel)
  await screen.findByText('triton')
  await fireEvent.click(screen.getByRole('button', { name: /clear/i }))
  const alert = await screen.findByRole('alert')
  expect(alert.textContent).toContain("Couldn't clear that: A job is running.")
  expect(container.querySelector('details')?.hasAttribute('open')).toBe(false)
})

it('clear asks with the cost and runs once', async () => {
  const { confirmDialog } = await import('./confirm.svelte')
  listing.caches = [cache('triton')]
  clearCache.mockReset()
  clearCache.mockResolvedValue({})
  vi.mocked(confirmDialog).mockReset()
  vi.mocked(confirmDialog).mockResolvedValue(true)
  render(CachesPanel)
  await screen.findByText('triton')
  const button = screen.getByRole('button', { name: /clear the triton/i })
  await fireEvent.click(button)
  await fireEvent.click(button)
  await vi.waitFor(() => expect(clearCache).toHaveBeenCalledTimes(1))
  expect(vi.mocked(confirmDialog).mock.calls[0]).toEqual([
    '1 KB, 1 file. It is read again.',
    {
      title: 'Clear the triton cache at /cache/triton?',
      confirmLabel: 'Clear',
      cancelLabel: undefined,
      danger: false,
    },
  ])
})
