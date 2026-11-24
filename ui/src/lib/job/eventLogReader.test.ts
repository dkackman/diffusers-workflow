import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createReadSpacer } from './eventLogReader'

beforeEach(() => vi.useFakeTimers())
afterEach(() => vi.useRealTimers())

describe('createReadSpacer', () => {
  it('runs the first ask at once', () => {
    const run = vi.fn()
    createReadSpacer(1000, run).request()
    expect(run).toHaveBeenCalledTimes(1)
  })

  it('folds asks inside the gap into one trailing run', () => {
    const run = vi.fn()
    const spacer = createReadSpacer(1000, run)
    spacer.request()
    for (let i = 0; i < 10; i++) {
      vi.advanceTimersByTime(50)
      spacer.request()
    }
    expect(run).toHaveBeenCalledTimes(1)
    vi.advanceTimersByTime(499)
    expect(run).toHaveBeenCalledTimes(1)
    vi.advanceTimersByTime(1)
    expect(run).toHaveBeenCalledTimes(2)
  })

  it('schedules nothing without an ask', () => {
    const run = vi.fn()
    const spacer = createReadSpacer(1000, run)
    spacer.request()
    vi.advanceTimersByTime(10_000)
    expect(run).toHaveBeenCalledTimes(1)
  })

  it('cancel drops the waiting run', () => {
    const run = vi.fn()
    const spacer = createReadSpacer(1000, run)
    spacer.request()
    spacer.request()
    spacer.cancel()
    vi.advanceTimersByTime(5000)
    expect(run).toHaveBeenCalledTimes(1)
    spacer.request()
    expect(run).toHaveBeenCalledTimes(2)
  })
})
