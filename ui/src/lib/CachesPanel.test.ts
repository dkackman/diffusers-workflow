import { cleanup, render, screen } from '@testing-library/svelte'
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
vi.mock('./api', () => ({ api: { listCaches: () => listCaches() } }))
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
