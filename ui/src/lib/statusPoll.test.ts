import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { pollStatus, STATUS_POLL_MS } from './statusPoll'
import app from '../App.svelte?raw'

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
  expect(app).toMatch(/return pollStatus\(refreshStatus\)/)
  // and the shell keeps no other timer of its own for it
  expect(app).not.toMatch(/\bpoll\(refreshStatus/)
})
