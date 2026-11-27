import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import ActivityPage from './ActivityPage.svelte'
import { connection } from '../connection.svelte'
import { api } from '../test/apiMock'
import type { HealthInfo, MemoryInfo } from '../types'
import { quiet, useQuietClock } from '../test/quiet'

useQuietClock()

const download = (over: Record<string, unknown> = {}) => ({
  id: 'd1',
  repo_id: 'org/big-model',
  status: 'downloading',
  downloaded: 1024 ** 3,
  total: 4 * 1024 ** 3,
  error: null,
  finished_at: null,
  ...over,
})
const state = vi.hoisted(() => ({
  downloads: [] as unknown[],
  diffusers: 'idle',
  dw: 'idle',
}))
vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))
api.server.mockImplementation(() =>
  Promise.resolve({ hostname: 'lem', version: '0.4.0', device: 'cuda' }),
)
api.listDownloads.mockImplementation(() =>
  Promise.resolve({ downloads: state.downloads }),
)
api.cancelDownload.mockImplementation(() => Promise.resolve({}))
api.diffusersStatus.mockImplementation(() =>
  Promise.resolve({ version: '0.36.0', commit: null, status: state.diffusers }),
)
api.dwStatus.mockImplementation(() =>
  Promise.resolve({
    version: '0.23.0',
    status: state.dw,
    newer: false,
    managed: true,
  }),
)

afterEach(() => {
  cleanup()
  state.downloads = []
  state.diffusers = 'idle'
  state.dw = 'idle'
  connection.health = null
  connection.memory = null
  api.cancelDownload.mockClear()
})

/** Render and return the page's collapsed text. */
async function show() {
  const { container } = render(ActivityPage)
  await quiet()
  return flat(container)
}
const flat = (el: Element) => (el.textContent ?? '').replace(/\s+/g, ' ')

it('sets the tab title to Activity, with host, version and device beside it', async () => {
  render(ActivityPage)
  await quiet()
  expect(document.title).toBe('Activity · diffusers-workflow')
  expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Activity')
  expect(
    screen.getByText('lem · v0.4.0 · cuda', { selector: '.secondary' }),
  ).toBeTruthy()
})

it('has the queue and none of the Connect or System panels', async () => {
  const { container } = render(ActivityPage)
  await quiet()
  expect(
    screen.getByRole('heading', { level: 2, name: 'All jobs' }),
  ).toBeTruthy()
  const text = flat(container)
  expect(text).not.toContain('Authentication')
  expect(text).not.toContain('Runtime')
  expect(text).not.toContain('Update from GitHub')
})

it('shows an active download with a Cancel button', async () => {
  state.downloads = [download()]
  const { container } = render(ActivityPage)
  await quiet()
  expect(flat(container)).toContain('org/big-model')
  expect(
    screen.getByRole('button', { name: 'cancel downloading org/big-model' }),
  ).toBeTruthy()
})

it('shows no download line when none is active', async () => {
  const { container } = render(ActivityPage)
  await quiet()
  expect(container.querySelector('.dl')).toBeNull()
})

it('links to System while a dw or diffusers update runs', async () => {
  state.dw = 'running'
  render(ActivityPage)
  await quiet()
  const link = screen.getByRole('link', { name: 'dw / diffusers updating…' })
  expect(link.getAttribute('href')).toBe('#/server/system')
})

it('shows no update line when nothing is updating', async () => {
  render(ActivityPage)
  await quiet()
  expect(screen.queryByText(/updating…/)).toBeNull()
})

it('lists each worker with its card, VRAM and a link to its job', async () => {
  connection.health = {
    status: 'ok',
    worker_alive: true,
    current_job: 'job-a',
    queued: 0,
    workers: [
      {
        device: 'cuda:0',
        name: 'cuda:0 NVIDIA GeForce RTX 3090',
        vram_gb: 24,
        current_job: 'job-a',
        alive: true,
        host_memory_rss_mb: 2048,
      },
      {
        device: 'cuda:1',
        name: 'cuda:1 NVIDIA GeForce RTX 3090',
        vram_gb: 24,
        current_job: 'job-b',
        alive: true,
      },
    ],
  } as unknown as HealthInfo
  connection.state = 'ok'
  const { container } = render(ActivityPage)
  await quiet()
  const text = (container.textContent ?? '').replace(/\s+/g, ' ')
  expect(text).toContain('cuda:0 NVIDIA GeForce RTX 3090')
  expect(text).toContain('cuda:1 NVIDIA GeForce RTX 3090')
  expect(text).toContain('2.0 GB RSS')
  const hrefs = [...container.querySelectorAll('li a')].map((a) =>
    a.getAttribute('href'),
  )
  expect(hrefs).toEqual(['#/ws/default/jobs/job-a', '#/ws/default/jobs/job-b'])
})

it('shows memory for every card, not just the first', async () => {
  const card = (device: string, allocated: number) => ({
    device,
    live: device === 'cuda:0',
    stale: device !== 'cuda:0',
    reason: device === 'cuda:0' ? null : 'job_running',
    age_seconds: device === 'cuda:0' ? 0 : 12,
    info: {
      gpu_available: true,
      gpu_device_name: 'NVIDIA GeForce RTX 3090',
      gpu_memory_allocated_mb: allocated,
      gpu_memory_total_mb: 24576,
    },
  })
  const first = card('cuda:0', 1024)
  connection.memory = {
    ...first,
    workers: [first, card('cuda:1', 18636.8)],
  } as unknown as MemoryInfo
  const text = await show()
  expect(text).toContain('cuda:0 1.0 / 24.0 GB')
  expect(text).toContain('cuda:1 18.2 / 24.0 GB 12s ago')
})

it('shows each worker pid and per-card free and reserved memory', async () => {
  connection.health = {
    status: 'ok',
    worker_alive: true,
    current_job: null,
    queued: 0,
    workers: [
      { device: 'cuda:0', name: 'cuda:0', alive: true, pid: 4242 },
      { device: 'cuda:1', name: 'cuda:1', alive: false },
    ],
  } as unknown as HealthInfo
  connection.state = 'ok'
  const entry = {
    device: 'cuda:0',
    live: true,
    stale: false,
    reason: null,
    age_seconds: 0,
    info: {
      gpu_available: true,
      gpu_memory_allocated_mb: 1024,
      gpu_memory_total_mb: 24576,
      gpu_memory_free_mb: 20480,
      gpu_memory_reserved_mb: 2048,
    },
  }
  connection.memory = { ...entry, workers: [entry] } as unknown as MemoryInfo
  const text = await show()
  expect(text).toContain('pid 4242')
  expect(text.match(/pid /g)).toHaveLength(1)
  expect(text).toContain('20.0 GB free')
  expect(text).toContain('2.0 GB reserved')
})
