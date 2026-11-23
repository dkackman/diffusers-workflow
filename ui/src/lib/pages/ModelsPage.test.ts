import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import ModelsPage from './ModelsPage.svelte'
import { formatDay } from '../format'
import { connection } from '../connection.svelte'
import { NetworkError } from '../apiErrors'
import type { ModelCache, ModelRepo } from '../types'
import { feed } from '../test/feedMock'
import { workspace } from '../workspaceState.svelte'
import { confirmDialog } from '../confirm.svelte'

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
  // how many times the diffusers status was read
  diffusersReads: 0,
  // the diffusers status the page reads; null stays pending
  diffusers: null as unknown,
}))
const calls = vi.hoisted(() => ({
  updateDiffusers: vi.fn(() => Promise.resolve({})),
  startDownload: vi.fn(() => new Promise(() => {})),
  deleteModel: vi.fn(() => Promise.resolve({})),
}))
vi.mock('../confirm.svelte', () => ({ confirmDialog: vi.fn() }))

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
    diffusersStatus: () => {
      cache.diffusersReads++
      if (cache.error) return Promise.reject(cache.error)
      return cache.diffusers
        ? Promise.resolve(cache.diffusers)
        : new Promise(() => {})
    },
    ...calls,
  },
}))
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

afterEach(() => {
  cleanup()
  cache.repos = []
  cache.error = null
  cache.diffusersReads = 0
  cache.diffusers = null
  vi.mocked(confirmDialog).mockReset()
  Object.values(calls).forEach((c) => c.mockClear())
})

async function lastUsedCell(repoId: string): Promise<HTMLElement> {
  render(ModelsPage)
  const name = await waitFor(() => screen.getByText(repoId))
  const row = name.closest('tr') as HTMLElement
  return row.querySelector('td.lastused') as HTMLElement
}

it('sets the tab title to Models', async () => {
  render(ModelsPage)
  await waitFor(() =>
    expect(document.title).toBe('Models · diffusers-workflow'),
  )
  expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Models')
})

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
  cache.error = new NetworkError('fetch failed')
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

it('ModelsPage reloads its cache list on a models move', async () => {
  cache.repos = [repo('org/first', DW_USE, null)]
  render(ModelsPage)
  await screen.findByText('org/first')
  expect(feed.watched(workspace.current)).toContain('models')
  cache.repos = [
    repo('org/first', DW_USE, null),
    repo('org/fetched', null, null),
  ]
  feed.move(workspace.current, 'models')
  await screen.findByText('org/fetched')
  // Quiet: the list never went back to loading
  expect(screen.getByText('org/first')).toBeTruthy()
})

it('a models move reads the diffusers status again', async () => {
  render(ModelsPage)
  await waitFor(() => expect(cache.diffusersReads).toBe(1))
  // An update started by another tab or an agent bumps models
  feed.move(workspace.current, 'models')
  await waitFor(() => expect(cache.diffusersReads).toBe(2))
})

it('Update diffusers asks first and runs only on yes', async () => {
  cache.diffusers = { version: '0.30', commit: null, status: 'idle' }
  render(ModelsPage)
  const button = await screen.findByRole('button', {
    name: /Update from GitHub/,
  })
  vi.mocked(confirmDialog).mockResolvedValueOnce(false)
  await fireEvent.click(button)
  await waitFor(() => expect(confirmDialog).toHaveBeenCalledTimes(1))
  expect(calls.updateDiffusers).not.toHaveBeenCalled()
  vi.mocked(confirmDialog).mockResolvedValueOnce(true)
  await fireEvent.click(button)
  await waitFor(() => expect(calls.updateDiffusers).toHaveBeenCalledTimes(1))
  expect(vi.mocked(confirmDialog).mock.calls[1]).toEqual([
    "This changes the server's Python environment and restarts the idle worker.",
    {
      title: 'Update diffusers from GitHub?',
      confirmLabel: 'Update diffusers',
      cancelLabel: undefined,
      danger: false,
    },
  ])
})

it('a second click while starting a download sends one request', async () => {
  render(ModelsPage)
  const input = await screen.findByPlaceholderText(
    'download a model… (org/name)',
  )
  await fireEvent.input(input, { target: { value: 'org/model' } })
  const button = screen.getByRole('button', { name: /Download/ })
  await fireEvent.click(button)
  await fireEvent.click(button)
  await fireEvent.submit(input.closest('form') as HTMLFormElement)
  expect(calls.startDownload).toHaveBeenCalledTimes(1)
  expect(confirmDialog).not.toHaveBeenCalled()
})

it('deleting a model asks with its size', async () => {
  cache.repos = [repo('org/big', DW_USE, null)]
  vi.mocked(confirmDialog).mockResolvedValue(true)
  render(ModelsPage)
  await screen.findByText('org/big')
  await fireEvent.click(
    screen.getByRole('button', { name: 'delete org/big from the cache' }),
  )
  await waitFor(() => expect(calls.deleteModel).toHaveBeenCalledWith('org/big'))
  expect(vi.mocked(confirmDialog).mock.calls[0]).toEqual([
    'The next workflow that needs it downloads it again.',
    {
      title: 'Delete org/big (0.0 GB) from the model cache?',
      confirmLabel: 'Delete model',
      cancelLabel: undefined,
      danger: false,
    },
  ])
})
