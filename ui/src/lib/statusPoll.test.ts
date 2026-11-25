import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import {
  checkNow,
  pollStatus,
  startStatusPoll,
  STATUS_POLL_MS,
} from './statusPoll'
import { connection } from './connection.svelte'
import { attentionBackends } from './attentionBackends.svelte'

function setHidden(hidden: boolean) {
  Object.defineProperty(document, 'visibilityState', {
    configurable: true,
    get: () => (hidden ? 'hidden' : 'visible'),
  })
  document.dispatchEvent(new Event('visibilitychange'))
}
beforeEach(() => vi.useFakeTimers())
afterEach(() => {
  setHidden(false)
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

it('the status poll keeps firing while the tab is hidden', () => {
  const refresh = vi.fn()
  const stop = pollStatus(refresh)
  expect(refresh).toHaveBeenCalledTimes(1)
  setHidden(true)
  vi.advanceTimersByTime(STATUS_POLL_MS * 3)
  expect(refresh).toHaveBeenCalledTimes(4)
  stop()
})

it('started in a hidden tab, it fires at once', () => {
  setHidden(true)
  const refresh = vi.fn()
  const stop = pollStatus(refresh)
  expect(refresh).toHaveBeenCalledTimes(1)
  stop()
})

it('starting it reads the status at once, then on each tick, hidden', async () => {
  const fetch = vi.fn(() => answer(200, { workers: [], devices: [] }))
  vi.stubGlobal('fetch', fetch)
  setHidden(true)
  const stop = startStatusPoll()
  // memory and health, each read apart
  expect(fetch).toHaveBeenCalledTimes(2)
  await vi.advanceTimersByTimeAsync(STATUS_POLL_MS)
  expect(fetch).toHaveBeenCalledTimes(4)
  stop()
  await vi.advanceTimersByTimeAsync(STATUS_POLL_MS)
  expect(fetch).toHaveBeenCalledTimes(4)
})

function answer(status: number, body: unknown) {
  return Promise.resolve({
    ok: status < 400,
    status,
    statusText: 'x',
    headers: new Headers(),
    json: () => Promise.resolve(body),
  })
}

it('a failed health read leaves memory and health null and reports offline', async () => {
  vi.stubGlobal('fetch', (url: string) =>
    url.endsWith('/api/health')
      ? answer(200, { status: 'ok', workers: [], attention_backends: ['sdpa'] })
      : answer(200, { devices: [] }),
  )
  await checkNow()
  expect(connection.health).not.toBeNull()
  expect(connection.memory).not.toBeNull()
  expect(attentionBackends.list).toEqual(['sdpa'])

  vi.stubGlobal('fetch', () => Promise.reject(new TypeError('Failed to fetch')))
  await checkNow()
  expect(connection.health).toBeNull()
  expect(connection.memory).toBeNull()
  expect(connection.state).toBe('offline')
  // the last known backends stay
  expect(attentionBackends.list).toEqual(['sdpa'])
})

it('memory failing alone keeps health', async () => {
  vi.stubGlobal('fetch', (url: string) =>
    url.endsWith('/api/health')
      ? answer(200, { status: 'ok', workers: [] })
      : answer(503, { detail: 'worker unreachable' }),
  )
  await checkNow()
  expect(connection.health).not.toBeNull()
  expect(connection.memory).toBeNull()
  expect(connection.state).toBe('ok')
})
