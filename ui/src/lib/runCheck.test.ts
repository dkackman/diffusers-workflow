import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { CHECK_AFTER_MS, RunCheck, SIZES_AFTER_MS } from './runCheck.svelte'
import type { ValidationResult } from './types'

const answer = (overrides: Partial<ValidationResult> = {}): ValidationResult =>
  ({
    valid: true,
    error: null,
    errors: [],
    warnings: [],
    ...overrides,
  }) as ValidationResult

beforeEach(() => vi.useFakeTimers())
afterEach(() => vi.useRealTimers())

it('checks a new workflow at once, and with sizes only after the idle time', async () => {
  const ask = vi.fn().mockResolvedValue(answer())
  const check = new RunCheck(ask)
  check.changed('wf', {})
  await vi.advanceTimersByTimeAsync(0)
  expect(ask).toHaveBeenCalledTimes(1)
  expect(ask.mock.calls[0][0]).toBe(false)
  expect(check.verdict).toEqual({
    tone: 'good',
    text: 'a run would be admitted',
  })
  expect(check.checking).toBe(false)

  await vi.advanceTimersByTimeAsync(SIZES_AFTER_MS)
  expect(ask).toHaveBeenCalledTimes(2)
  expect(ask.mock.calls[1][0]).toBe(true)
})

it('waits for a pause after a changed value, and not at all for the same one', async () => {
  const ask = vi.fn().mockResolvedValue(answer())
  const check = new RunCheck(ask)
  check.changed('wf', {})
  await vi.advanceTimersByTimeAsync(0)
  check.changed('wf', {})
  check.changed('wf', { steps: '4' })
  check.changed('wf', { steps: '40' })
  await vi.advanceTimersByTimeAsync(CHECK_AFTER_MS - 1)
  expect(ask).toHaveBeenCalledTimes(1)
  await vi.advanceTimersByTimeAsync(1)
  expect(ask).toHaveBeenCalledTimes(2)
})

it('drops an answer from before the latest change', async () => {
  let release: (value: ValidationResult) => void = () => {}
  const ask = vi
    .fn()
    .mockImplementationOnce(
      () => new Promise<ValidationResult>((resolve) => (release = resolve)),
    )
    .mockResolvedValue(answer({ warnings: ['slow'] }))
  const check = new RunCheck(ask)
  check.changed('wf', {})
  await vi.advanceTimersByTimeAsync(0)
  check.changed('wf', { steps: '4' })
  await vi.advanceTimersByTimeAsync(CHECK_AFTER_MS)
  release(answer({ valid: false }))
  await vi.advanceTimersByTimeAsync(0)
  expect(check.verdict?.tone).toBe('warn')
})

it('names a refused check, and says nothing of an aborted one', async () => {
  const ask = vi.fn().mockRejectedValue(new Error('unknown workflow'))
  const check = new RunCheck(ask)
  check.changed('wf', {})
  await vi.advanceTimersByTimeAsync(0)
  expect(check.error).toBe('unknown workflow')
  expect(check.verdict).toEqual({
    tone: 'bad',
    text: 'could not check this run',
  })

  ask.mockRejectedValue(new DOMException('aborted', 'AbortError'))
  check.changed('other', {})
  expect(check.error).toBe('')
  await vi.advanceTimersByTimeAsync(0)
  expect(check.error).toBe('')
  expect(check.verdict).toBeNull()
})

it('asks nothing more once cleared or stopped', async () => {
  const ask = vi.fn().mockResolvedValue(answer())
  const check = new RunCheck(ask)
  check.changed('wf', {})
  check.clear()
  await vi.advanceTimersByTimeAsync(SIZES_AFTER_MS)
  expect(ask).not.toHaveBeenCalled()

  // the same workflow again is a first check again
  check.changed('wf', {})
  check.stop()
  await vi.advanceTimersByTimeAsync(SIZES_AFTER_MS)
  expect(ask).not.toHaveBeenCalled()
})
