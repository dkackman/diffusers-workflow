import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'

vi.mock('./lib/statusPoll', () => ({ startStatusPoll: () => () => {} }))
vi.mock('./lib/api', async (original) => {
  const real = await original<typeof import('./lib/api')>()
  const never = () => new Promise(() => {})
  return { ...real, api: new Proxy({}, { get: () => never }) }
})
vi.mock('./lib/toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

import App from './App.svelte'
import { connection } from './lib/connection.svelte'
import { holdLayer } from './lib/ui/layers.svelte'
import type { HealthInfo, MemoryInfo } from './lib/types'

function at(hash: string) {
  location.hash = hash
  window.dispatchEvent(new HashChangeEvent('hashchange'))
}

const worker = (current_job: string | null) => ({
  device: 'cuda:0',
  name: 'cuda:0 GPU',
  vram_gb: 24,
  current_job,
  alive: true,
})
const health = (...jobs: string[]): HealthInfo =>
  ({
    status: 'ok',
    version: '1',
    hostname: 'h',
    device: 'cuda',
    mcp: false,
    queued: 0,
    worker_alive: true,
    current_job: jobs[0] ?? null,
    workers: jobs.length ? jobs.map(worker) : [worker(null)],
  }) as HealthInfo
const memory = {
  device: 'cuda:0',
  live: true,
  age_seconds: 1,
  info: {
    gpu_available: true,
    gpu_memory_total_mb: 24000,
    gpu_memory_allocated_mb: 6000,
    gpu_memory_free_mb: 18000,
  },
  workers: [],
} as unknown as MemoryInfo

beforeEach(() => {
  vi.spyOn(window, 'scrollTo').mockImplementation(() => {})
  localStorage.clear()
  at('#/zzz')
  connection.state = 'ok'
  connection.health = null
  connection.memory = null
})
afterEach(() => {
  cleanup()
  document.body.innerHTML = ''
})

it('a navigation closes the status and token popovers', async () => {
  render(App)
  const status = screen.getByRole('button', { name: /status|idle/ })
  const token = screen.getByRole('button', { name: 'API token' })
  await fireEvent.click(status)
  await fireEvent.click(token)
  expect(status.getAttribute('aria-expanded')).toBe('true')
  expect(token.getAttribute('aria-expanded')).toBe('true')
  at('#/ws/default/jobs')
  await waitFor(() => {
    expect(status.getAttribute('aria-expanded')).toBe('false')
    expect(token.getAttribute('aria-expanded')).toBe('false')
  })
})

it('the help button opens the shortcuts', async () => {
  render(App)
  await fireEvent.click(
    screen.getByRole('button', { name: 'Keyboard shortcuts' }),
  )
  expect(await screen.findByText("search this page's list")).toBeTruthy()
})

function addSearch() {
  const head = document.createElement('div')
  head.className = 'pagehead'
  head.innerHTML = '<input class="filter" aria-label="find">'
  document.querySelector('main')!.appendChild(head)
  return head.querySelector('input')!
}

it("slash focuses the page's search", async () => {
  render(App)
  const search = addSearch()
  const event = new KeyboardEvent('keydown', {
    key: '/',
    bubbles: true,
    cancelable: true,
  })
  document.body.dispatchEvent(event)
  expect(document.activeElement).toBe(search)
  expect(event.defaultPrevented).toBe(true)
})

it('slash in a field types a slash', async () => {
  render(App)
  const search = addSearch()
  const other = document.createElement('input')
  document.body.appendChild(other)
  other.focus()
  const event = new KeyboardEvent('keydown', {
    key: '/',
    bubbles: true,
    cancelable: true,
  })
  other.dispatchEvent(event)
  expect(event.defaultPrevented).toBe(false)
  expect(document.activeElement).toBe(other)
  expect(search).not.toBe(document.activeElement)
})

it('slash under an open overlay does nothing', () => {
  render(App)
  const search = addSearch()
  const release = holdLayer()
  document.body.dispatchEvent(
    new KeyboardEvent('keydown', { key: '/', bubbles: true }),
  )
  release()
  expect(document.activeElement).not.toBe(search)
})

it('slash with no search on the page leaves the key alone', () => {
  render(App)
  const event = new KeyboardEvent('keydown', {
    key: '/',
    bubbles: true,
    cancelable: true,
  })
  document.body.dispatchEvent(event)
  expect(event.defaultPrevented).toBe(false)
})

it("the meter's label starts with its state word", () => {
  connection.memory = memory
  connection.health = health()
  render(App)
  expect(
    screen
      .getByRole('button', { name: /server & worker status/ })
      .getAttribute('aria-label'),
  ).toBe('idle, server & worker status, 5.9/23.4 GB')
})

it('one running job is said once', () => {
  connection.memory = memory
  connection.health = health('job-a')
  render(App)
  const meter = screen.getByRole('button', { name: /server & worker status/ })
  expect(meter.getAttribute('aria-label')).toMatch(/^1 job running, server/)
  expect(meter.textContent).not.toMatch(/running|idle/)
  expect(screen.getAllByText(/running/)).toHaveLength(1)
})

it("the meter's name keeps the running count", () => {
  connection.memory = memory
  connection.health = health('job-a', 'job-b')
  render(App)
  expect(
    screen
      .getByRole('button', { name: /server & worker status/ })
      .getAttribute('aria-label'),
  ).toBe('2 jobs running, server & worker status, 5.9/23.4 GB')
})

it('several running jobs: the meter label counts them', () => {
  connection.memory = memory
  connection.health = health('job-a', 'job-b', 'job-c')
  render(App)
  const meter = screen.getByRole('button', { name: /server & worker status/ })
  expect(meter.getAttribute('aria-label')).toMatch(/^3 jobs running, server/)
  expect(meter.textContent).not.toMatch(/idle/)
})

it('no meter: N running and status labels', () => {
  connection.health = health('job-a', 'job-b')
  render(App)
  expect(
    screen.getByRole('button', {
      name: '2 jobs running - server & worker status',
    }),
  ).toBeTruthy()
  cleanup()
  connection.health = health()
  render(App)
  expect(
    screen.getByRole('button', { name: 'server & worker status' }),
  ).toBeTruthy()
})
