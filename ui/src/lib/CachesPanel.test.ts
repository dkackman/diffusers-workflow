import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import CachesPanel from './CachesPanel.svelte'
import { feed } from './test/feedMock'
import { workspace } from './workspaceState.svelte'
import { formatExact, formatMtime } from './format'

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

it('a disabled cache does not warn about disk', async () => {
  listing.caches = [{ ...cache('triton'), enabled: false, low_disk: true }]
  render(CachesPanel)
  await screen.findByText('triton')
  expect(screen.queryByText(/free-space floor/)).toBeNull()
})

it('an enabled cache under its floor warns', async () => {
  listing.caches = [{ ...cache('triton'), enabled: true, low_disk: true }]
  render(CachesPanel)
  expect(await screen.findByText(/free-space floor/)).toBeTruthy()
})

it('a cache trash button is not drawn in danger', async () => {
  listing.caches = [cache('triton')]
  render(CachesPanel)
  await screen.findByText('triton')
  const button = screen.getByRole('button', { name: 'clear the triton cache' })
  expect(button.classList.contains('danger')).toBe(false)
})

it("a cache's last use and each entry's created and last hit are exact, to the second on hover, and a dash when unknown", async () => {
  const entry = {
    id: 'a'.repeat(64),
    component: 'transformer',
    model: 'org/model',
    subfolder: null,
    bytes: 1024,
    files: 1,
    created: 1700000000,
    last_hit: 1700000600,
    unreadable: false,
  }
  listing.caches = [
    { ...cache('quantized'), last_used: 1700001200, entries: [entry] },
    {
      ...cache('triton'),
      entries: [
        { ...entry, id: 'b'.repeat(64), created: null, last_hit: null },
      ],
    },
  ]
  const { container } = render(CachesPanel)
  await screen.findByText('quantized')
  const used = container.querySelector('tbody time') as HTMLElement
  expect(used.textContent?.trim()).toBe(formatMtime(1700001200))
  expect(used.getAttribute('title')).toBe(formatExact(1700001200))
  await fireEvent.click(screen.getAllByRole('button', { name: '1 entry' })[0])
  const cells = [
    ...container.querySelectorAll('.entrytable tbody tr'),
  ][0].querySelectorAll('td.muted')
  expect([...cells].map((td) => td.textContent?.trim())).toEqual([
    formatMtime(1700000000),
    formatMtime(1700000600),
  ])
  expect(
    [...cells].map((td) => td.querySelector('time')?.getAttribute('title')),
  ).toEqual([formatExact(1700000000), formatExact(1700000600)])
  await fireEvent.click(screen.getAllByRole('button', { name: '1 entry' })[1])
  const unknown = [
    ...container.querySelectorAll('.entrytable'),
  ][1].querySelectorAll('tbody td.muted')
  expect([...unknown].map((td) => td.textContent?.trim())).toEqual(['—', '—'])
})
