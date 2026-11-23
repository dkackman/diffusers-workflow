import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import ModelsPage from './ModelsPage.svelte'
import { formatDay } from '../format'
import { connection } from '../connection.svelte'
import type { ModelCache, ModelRepo } from '../types'

// The "last used" column (#1445 stage B): a dw job's start time when the
// server has one, else the files' atime marked "(disk)", else a dash

const DW_USE = 1_763_000_000
const ATIME = 1_760_000_000

const repo = (
  repo_id: string,
  last_job_use: number | null,
  last_accessed: number | null,
): ModelRepo => ({
  repo_id,
  repo_type: 'model',
  size_on_disk: 1024,
  nb_files: 3,
  last_accessed,
  last_job_use,
  last_modified: null,
  revisions: [],
})

const cache = vi.hoisted(() => ({
  repos: [] as ModelRepo[],
  // what the cache listing fails with, while it fails
  error: null as Error | null,
}))

vi.mock('../api', () => ({
  api: {
    listModels: () =>
      cache.error
        ? Promise.reject(cache.error)
        : Promise.resolve({
            cache_dir: '/hub',
            disk_free: null,
            disk_total: null,
            size_on_disk: 0,
            warnings: [],
            repos: cache.repos,
          } satisfies ModelCache),
    listDownloads: () =>
      cache.error
        ? Promise.reject(cache.error)
        : Promise.resolve({ downloads: [] }),
    diffusersStatus: () =>
      cache.error ? Promise.reject(cache.error) : new Promise(() => {}),
  },
}))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

afterEach(() => {
  cleanup()
  cache.repos = []
  cache.error = null
})

async function lastUsedCell(repoId: string): Promise<HTMLElement> {
  render(ModelsPage)
  const name = await waitFor(() => screen.getByText(repoId))
  const row = name.closest('tr') as HTMLElement
  return row.querySelector('td.lastused') as HTMLElement
}

it('shows the dw job date with no disk mark when last_job_use is set', async () => {
  cache.repos = [repo('org/used', DW_USE, ATIME)]
  const cell = await lastUsedCell('org/used')
  expect(cell.textContent?.trim()).toBe(formatDay(DW_USE))
  expect(cell.textContent).not.toContain('(disk)')
  expect(cell.classList.contains('muted')).toBe(false)
})

it('falls back to the atime, muted and labelled "(disk)" with its tooltip', async () => {
  cache.repos = [repo('org/copied', null, ATIME)]
  const cell = await lastUsedCell('org/copied')
  expect(cell.textContent?.trim()).toBe(`${formatDay(ATIME)} (disk)`)
  expect(cell.classList.contains('muted')).toBe(true)
  expect(cell.getAttribute('title')).toBe(
    'No dw job has used this repo since tracking began. ' +
      "Date is the files' last access time, which copies and backups also bump.",
  )
})

it('shows a dash when neither date is known', async () => {
  cache.repos = [repo('org/unknown', null, null)]
  const cell = await lastUsedCell('org/unknown')
  expect(cell.textContent?.trim()).toBe('—')
  expect(cell.textContent).not.toContain('(disk)')
})

it('explains the column in the header tooltip', async () => {
  cache.repos = [repo('org/used', DW_USE, ATIME)]
  await lastUsedCell('org/used')
  const header = screen.getByText('last used')
  expect(header.getAttribute('title')).toMatch(/dw job/)
  expect(header.getAttribute('title')).toMatch(/\(disk\)/)
})

it('recovers on reconnect', async () => {
  cache.error = new TypeError('fetch failed')
  render(ModelsPage)
  await screen.findByText("Couldn't load the hub cache: Can't reach the server")
  // Offline, only the cache list says so: downloads and the diffusers
  // version fail quietly beside it
  expect(screen.getAllByText(/Couldn't load/)).toHaveLength(1)
  expect(screen.getAllByRole('alert')).toHaveLength(1)
  expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
  expect(screen.queryByText('fetch failed')).toBeNull()
  // The download form stays usable while the listing is down
  expect(
    screen.getByPlaceholderText('download a model… (org/name)'),
  ).toBeTruthy()
  cache.error = null
  cache.repos = [repo('org/back', DW_USE, null)]
  connection.epoch++
  await screen.findByText('org/back')
  expect(screen.queryByText(/Couldn't load/)).toBeNull()
})
