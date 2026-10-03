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
