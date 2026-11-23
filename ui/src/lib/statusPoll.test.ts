import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import {
  checkNow,
  pollStatus,
  startStatusPoll,
  STATUS_POLL_MS,
} from './statusPoll'
import { connection } from './connection.svelte'
import { attentionBackends } from './attentionBackends.svelte'
import app from '../App.svelte?raw'
import statusPoll from './statusPoll.ts?raw'

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

it('is the timer the app shell polls its status with', () => {
  expect(app).toMatch(/\$effect\(\(\) => startStatusPoll\(\)\)/)
  expect(statusPoll).toMatch(/return pollStatus\(checkNow\)/)
  // and the shell keeps no timer, and no status read, of its own for it
  expect(app).not.toMatch(/\bpoll\(|api\.(health|memory)\(/)
})

it('starting it reads the status at once', () => {
  const fetch = vi.fn(() => new Promise(() => {}))
  vi.stubGlobal('fetch', fetch)
  const stop = startStatusPoll()
  // memory and health, each read apart
  expect(fetch).toHaveBeenCalledTimes(2)
  stop()
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
