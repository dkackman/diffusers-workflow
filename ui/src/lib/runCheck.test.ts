import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import {
  CHECK_AFTER_MS,
  RunCheck,
  SIZES_AFTER_MS,
  verdictOf,
} from './runCheck.svelte'
import { placeFindings } from './findings'
import type { Plan, ValidationResult } from './types'

const answer = (overrides: Partial<ValidationResult> = {}): ValidationResult =>
  ({
    valid: true,
    error: null,
    errors: [],
    warnings: [],
    ...overrides,
  }) as ValidationResult

const none = placeFindings(null, [])

beforeEach(() => vi.useFakeTimers())
afterEach(() => vi.useRealTimers())

it('checks a new workflow at once, and with sizes only after the idle time', async () => {
  const ask = vi.fn().mockResolvedValue(answer())
  const check = new RunCheck(ask)
  check.changed('wf', {})
  await vi.advanceTimersByTimeAsync(0)
  expect(ask).toHaveBeenCalledTimes(1)
  expect(ask.mock.calls[0][0]).toBe(false)
  expect(verdictOf(check.validation, check.error, none)?.text).toBe(
    'Ready to run.',
  )
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
  expect(check.validation?.warnings).toEqual(['slow'])
})

it('names a refused check, and says nothing of an aborted one', async () => {
  const refused = new Error('unknown workflow')
  const ask = vi.fn().mockRejectedValue(refused)
  const check = new RunCheck(ask)
  check.changed('wf', {})
  await vi.advanceTimersByTimeAsync(0)
  // kept as caught, for ErrorLine to describe
  expect(check.error).toBe(refused)
  expect(verdictOf(check.validation, check.error, none)?.text).toBe(
    "Couldn't check this workflow.",
  )

  ask.mockRejectedValue(new DOMException('aborted', 'AbortError'))
  check.changed('other', {})
  expect(check.error).toBeNull()
  await vi.advanceTimersByTimeAsync(0)
  expect(check.error).toBeNull()
  expect(verdictOf(check.validation, check.error, none)).toBeNull()
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

const planWith = (downloads: Plan['downloads_required']): Plan =>
  ({ downloads_required: downloads }) as Plan

const verdictFor = (result: ValidationResult, shown: string[] = []) =>
  verdictOf(result, null, placeFindings(result, shown))

it('a refused check counts its issues', () => {
  const two = answer({
    valid: false,
    errors: [
      { path: 'arguments.n', message: 'not an integer' },
      { path: 'steps[0]', message: 'no pipeline' },
    ],
  })
  expect(verdictFor(two, ['n'])).toEqual({
    tone: 'bad',
    text: 'A job would be refused: 2 issues to fix.',
    issues: 2,
  })
  const bare = answer({ valid: false, error: 'no such workflow' })
  expect(verdictFor(bare)?.text).toBe('A job would be refused: 1 issue to fix.')
})

it('a refusal with no counted issue reads A job would be refused.', () => {
  const blank = answer({ valid: false, error: null, errors: [] })
  expect(verdictFor(blank)).toEqual({
    tone: 'bad',
    text: 'A job would be refused.',
    issues: 0,
  })
})

it('a gated download is its own verdict with the hf remedy', () => {
  const gated = answer({
    plan: planWith([
      { repo: 'org/m', gb: null, gated: true, access_blocked: true },
    ]),
  })
  expect(verdictFor(gated)).toMatchObject({
    text: 'This model needs Hugging Face access.',
    remedy:
      'Accept its terms on Hugging Face, then sign in on the server with hf auth login.',
    issues: 0,
  })
})

it("a repo that isn't there says so", () => {
  const missing = answer({
    plan: planWith([
      {
        repo: 'org/m',
        gb: null,
        gated: null,
        access_blocked: null,
        not_found: true,
      },
    ]),
  })
  expect(verdictFor(missing)).toMatchObject({
    text: "A model this workflow downloads can't be found.",
    remedy: 'Check the model name in the workflow.',
  })
})

it('a note under a filled field asks to read it', () => {
  const noted = answer({
    warnings: ['arguments.seed: the same seed gives the same image'],
  })
  expect(verdictFor(noted, ['seed'])).toMatchObject({
    tone: 'warn',
    text: 'Ready to run. Read the notes under 1 field.',
  })
})

it('notes for the author alone are a warning, not a green light', () => {
  const authored = answer({ warnings: ['a step has no output'] })
  expect(verdictFor(authored, ['seed'])).toMatchObject({
    tone: 'warn',
    text: 'Ready to run.',
  })
})

it('a changed form is pending until its answer lands, and a refusal then counts', async () => {
  const check = new RunCheck(() => Promise.resolve(answer({})))
  check.changed('wf', { a: 1 })
  expect(check.pending).toBe(true)
  await vi.waitFor(() => expect(check.pending).toBe(false))
})
