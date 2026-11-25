import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { poll, sleep } from './poll'

beforeEach(() => vi.useFakeTimers())
afterEach(() => vi.useRealTimers())

it('calls at once, then once per interval, and never after stopping', () => {
  const fn = vi.fn()
  const stop = poll(fn, 1000)
  expect(fn).toHaveBeenCalledTimes(1)
  vi.advanceTimersByTime(2500)
  expect(fn).toHaveBeenCalledTimes(3)
  stop()
  vi.advanceTimersByTime(5000)
  expect(fn).toHaveBeenCalledTimes(3)
})

it('can wait a full interval before the first call', () => {
  const fn = vi.fn()
  poll(fn, 1000, { immediate: false })
  expect(fn).not.toHaveBeenCalled()
  vi.advanceTimersByTime(1000)
  expect(fn).toHaveBeenCalledTimes(1)
})

it('sleeps', async () => {
  const done = vi.fn()
  sleep(500).then(done)
  await vi.advanceTimersByTimeAsync(499)
  expect(done).not.toHaveBeenCalled()
  await vi.advanceTimersByTimeAsync(1)
  expect(done).toHaveBeenCalled()
})

// The tab's visibility, as the browser would report it
function setHidden(hidden: boolean) {
  Object.defineProperty(document, 'visibilityState', {
    configurable: true,
    get: () => (hidden ? 'hidden' : 'visible'),
  })
  document.dispatchEvent(new Event('visibilitychange'))
}
afterEach(() => setHidden(false))

it('does not tick while the tab is hidden, and fires once on its return', () => {
  const fn = vi.fn()
  const stop = poll(fn, 1000)
  expect(fn).toHaveBeenCalledTimes(1)
  setHidden(true)
  vi.advanceTimersByTime(10_000)
  expect(fn).toHaveBeenCalledTimes(1)
  setHidden(false)
  expect(fn).toHaveBeenCalledTimes(2)
  // and the interval picks up again from there, once per interval
  vi.advanceTimersByTime(1000)
  expect(fn).toHaveBeenCalledTimes(3)
  stop()
})

it('started while hidden, waits for the tab to show', () => {
  setHidden(true)
  const fn = vi.fn()
  const stop = poll(fn, 1000)
  vi.advanceTimersByTime(5000)
  expect(fn).not.toHaveBeenCalled()
  setHidden(false)
  expect(fn).toHaveBeenCalledTimes(1)
  stop()
})

it('a stopped poll does not fire on the tab returning', () => {
  const fn = vi.fn()
  const stop = poll(fn, 1000)
  setHidden(true)
  stop()
  setHidden(false)
  vi.advanceTimersByTime(5000)
  expect(fn).toHaveBeenCalledTimes(1)
})

it('an opted-out poll keeps firing while the tab is hidden', () => {
  const fn = vi.fn()
  setHidden(true)
  const stop = poll(fn, 1000, { whileHidden: true })
  expect(fn).toHaveBeenCalledTimes(1)
  vi.advanceTimersByTime(3000)
  expect(fn).toHaveBeenCalledTimes(4)
  // and the tab returning does not add a call of its own
  setHidden(false)
  expect(fn).toHaveBeenCalledTimes(4)
  stop()
})

it('skips a tick while the previous call is still pending', async () => {
  let finish = () => {}
  const fn = vi.fn(() => new Promise<void>((resolve) => (finish = resolve)))
  const stop = poll(fn, 1000)
  expect(fn).toHaveBeenCalledTimes(1)
  await vi.advanceTimersByTimeAsync(3000)
  expect(fn).toHaveBeenCalledTimes(1)
  finish()
  await vi.advanceTimersByTimeAsync(1000)
  expect(fn).toHaveBeenCalledTimes(2)
  stop()
})

it('a call that rejects or throws does not stop the timer', async () => {
  const rejects = vi.fn(() => Promise.reject(new Error('503')))
  const throws = vi.fn(() => {
    throw new Error('boom')
  })
  const stops = [poll(rejects, 1000), poll(throws, 1000)]
  await vi.advanceTimersByTimeAsync(3000)
  expect(rejects).toHaveBeenCalledTimes(4)
  expect(throws).toHaveBeenCalledTimes(4)
  stops.forEach((stop) => stop())
})

it('once stops after its first call', () => {
  vi.useFakeTimers()
  const fn = vi.fn()
  const stop = poll(fn, 1000, { immediate: false, once: true })
  vi.advanceTimersByTime(999)
  expect(fn).not.toHaveBeenCalled()
  vi.advanceTimersByTime(1)
  expect(fn).toHaveBeenCalledTimes(1)
  vi.advanceTimersByTime(5000)
  expect(fn).toHaveBeenCalledTimes(1)
  stop()
})

it('once waits out a hidden tab, then fires one interval after its return', () => {
  const fn = vi.fn()
  const stop = poll(fn, 1000, { immediate: false, once: true })
  vi.advanceTimersByTime(500)
  setHidden(true)
  vi.advanceTimersByTime(10_000)
  expect(fn).not.toHaveBeenCalled()
  setHidden(false)
  expect(fn).not.toHaveBeenCalled()
  vi.advanceTimersByTime(1000)
  expect(fn).toHaveBeenCalledTimes(1)
  vi.advanceTimersByTime(5000)
  expect(fn).toHaveBeenCalledTimes(1)
  stop()
})
