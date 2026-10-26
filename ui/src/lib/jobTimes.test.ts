import { describe, expect, it } from 'vitest'
import { queueWait } from './jobTimes'

describe('queueWait', () => {
  it('is the time from submit to start for a job that started', () => {
    expect(
      queueWait({ created_at: 100, started_at: 160, status: 'succeeded' }, 999),
    ).toBe(60)
  })
  it('counts up to now for a job still queued', () => {
    expect(
      queueWait({ created_at: 100, started_at: null, status: 'queued' }, 400),
    ).toBe(300)
  })
  it('is null for a job cancelled before it started, or with no clock', () => {
    expect(
      queueWait(
        { created_at: 100, started_at: null, status: 'cancelled' },
        400,
      ),
    ).toBeNull()
    expect(
      queueWait({ created_at: null, started_at: null, status: 'queued' }, 400),
    ).toBeNull()
  })
})
