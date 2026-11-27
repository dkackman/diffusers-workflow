import { flushSync } from 'svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { ApiError } from '../apiErrors'
import { api, resetApi } from '../test/apiMock'
import { dryRun } from './dryRun.svelte'
import type { Ladder } from './ladderDraft'

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())

let teardowns: (() => void)[] = []
beforeEach(() => {
  resetApi()
  vi.useFakeTimers()
})
afterEach(() => {
  teardowns.forEach((t) => t())
  teardowns = []
  vi.useRealTimers()
})

/** A dry run made inside an effect root, as a component's setup makes one,
 * over a draft the test edits */
function make(first: Ladder, blocked: string[] = []) {
  const state = $state({ draft: first })
  let run!: ReturnType<typeof dryRun>
  teardowns.push(
    $effect.root(() => {
      run = dryRun(
        () => 'text_to_image',
        () => state.draft,
        () => blocked,
      )
    }),
  )
  flushSync()
  return { run, state }
}

const answer = (target: string) => ({
  tiers: { best: { target, skipped: [] } },
})

it('checks once, 300 ms after the last of two changes', async () => {
  api.checkPreference.mockResolvedValue(answer('a'))
  const { run, state } = make({ best: ['a'] }, ['models/flux-dev'])
  state.draft = { best: ['a', 'b'] }
  flushSync()
  await vi.advanceTimersByTimeAsync(100)
  state.draft = { best: ['b', 'a'], draft: [] }
  flushSync()
  await vi.advanceTimersByTimeAsync(299)
  expect(api.checkPreference).not.toHaveBeenCalled()
  await vi.advanceTimersByTimeAsync(1)
  expect(api.checkPreference).toHaveBeenCalledTimes(1)
  expect(api.checkPreference).toHaveBeenCalledWith('text_to_image', {
    ladder: { best: ['b', 'a'] },
    full: false,
    known_blocked: ['models/flux-dev'],
  })
  await vi.advanceTimersByTimeAsync(0)
  expect(run.answer).toEqual(answer('a'))
  expect(run.checking).toBe(false)
})

it('drops an answer older than the latest check sent', async () => {
  let first!: (v: unknown) => void
  let second!: (v: unknown) => void
  api.checkPreference
    .mockImplementationOnce(() => new Promise((r) => (first = r)))
    .mockImplementationOnce(() => new Promise((r) => (second = r)))
  const { run, state } = make({ best: ['a'] })
  await vi.advanceTimersByTimeAsync(300)
  state.draft = { best: ['b'] }
  flushSync()
  await vi.advanceTimersByTimeAsync(300)
  expect(api.checkPreference).toHaveBeenCalledTimes(2)
  expect(run.checking).toBe(true)
  second(answer('b'))
  await vi.advanceTimersByTimeAsync(0)
  first(answer('a'))
  await vi.advanceTimersByTimeAsync(0)
  expect(run.answer).toEqual(answer('b'))
  expect(run.checking).toBe(false)
})

it("drops an answer that lands during the next draft's quiet", async () => {
  // The first check answers after the draft moved on but before the second
  // is sent: it is the earlier draft's, and must not be shown against this one
  let first!: (v: unknown) => void
  api.checkPreference
    .mockImplementationOnce(() => new Promise((r) => (first = r)))
    .mockResolvedValue(answer('b'))
  const { run, state } = make({ best: ['a'] })
  await vi.advanceTimersByTimeAsync(300)
  state.draft = { best: ['b'] }
  flushSync()
  first(answer('a'))
  await vi.advanceTimersByTimeAsync(100)
  expect(api.checkPreference).toHaveBeenCalledTimes(1)
  expect(run.answer).toBeNull()
  expect(run.checking).toBe(true)
  await vi.advanceTimersByTimeAsync(200)
  expect(run.answer).toEqual(answer('b'))
  expect(run.checking).toBe(false)
})

it("holds a 422 as the draft's shape error", async () => {
  const refusal = new ApiError("tier best: 'x' is not a target", 422)
  api.checkPreference.mockRejectedValueOnce(refusal)
  const { run } = make({ best: ['x'] })
  await vi.advanceTimersByTimeAsync(300)
  expect(run.refused).toBe(refusal)
  expect(run.answer).toBeNull()
})

it('a failed check drops the last answer and says so', async () => {
  api.checkPreference
    .mockResolvedValueOnce(answer('a'))
    .mockRejectedValueOnce(new ApiError('server error', 500))
    .mockResolvedValueOnce(answer('c'))
  const { run, state } = make({ best: ['a'] })
  await vi.advanceTimersByTimeAsync(300)
  expect(run.answer).toEqual(answer('a'))
  expect(run.failed).toBe(false)
  state.draft = { best: ['b'] }
  flushSync()
  await vi.advanceTimersByTimeAsync(300)
  // the answer was the previous draft's: it no longer stands
  expect(run.answer).toBeNull()
  expect(run.refused).toBeNull()
  expect(run.failed).toBe(true)
  state.draft = { best: ['c'] }
  flushSync()
  await vi.advanceTimersByTimeAsync(300)
  expect(run.answer).toEqual(answer('c'))
  expect(run.failed).toBe(false)
})

it('sends nothing for a ladder with no model in it', async () => {
  const { run } = make({ best: [] })
  await vi.advanceTimersByTimeAsync(600)
  expect(api.checkPreference).not.toHaveBeenCalled()
  expect(run.answer).toBeNull()
})
