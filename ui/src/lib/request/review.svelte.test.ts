import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { item, request } from './fixtures'
import { ReviewCountdown, TICK_MS, Ticker } from './review.svelte'
import type { LedgerReview } from '../types'

let hidden = false
beforeEach(() => {
  vi.useFakeTimers()
  vi.setSystemTime(new Date('2026-11-21T12:00:00Z'))
  hidden = false
  vi.spyOn(document, 'visibilityState', 'get').mockImplementation(() =>
    hidden ? 'hidden' : 'visible',
  )
})
afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

// The server's deadline is a day off the browser's clock on purpose: the
// countdown must come from seconds_left, never from comparing with `until`
const review = (
  seconds_left: number,
  state: LedgerReview['state'] = 'holding',
): LedgerReview => ({ until: '2026-11-22T12:00:00+00:00', state, seconds_left })
const held = (seconds_left: number, state?: LedgerReview['state']) =>
  request([item('a', { review: review(seconds_left, state) }), item('b')])

it('counts down from the read plus seconds_left, not the deadline', async () => {
  const clock = new Ticker()
  const countdown = new ReviewCountdown(vi.fn(), clock)
  countdown.read(held(90))
  const release = countdown.hold()
  expect(countdown.left('a')).toBe(90)
  expect(countdown.left('b')).toBeNull()
  await vi.advanceTimersByTimeAsync(TICK_MS * 30)
  expect(countdown.left('a')).toBe(60)
  // a newer read re-anchors it
  countdown.read(held(45))
  expect(countdown.left('a')).toBe(45)
  release()
})

it('re-reads the request once when a countdown reaches zero', async () => {
  const refresh = vi.fn()
  const clock = new Ticker()
  const countdown = new ReviewCountdown(refresh, clock)
  countdown.read(held(3))
  const release = countdown.hold()
  await vi.advanceTimersByTimeAsync(TICK_MS * 2)
  expect(refresh).not.toHaveBeenCalled()
  await vi.advanceTimersByTimeAsync(TICK_MS)
  expect(countdown.left('a')).toBe(0)
  expect(refresh).toHaveBeenCalledTimes(1)
  // the re-read still says holding (the server's clock is a second behind):
  // the page does not ask again for the same deadline
  countdown.read(held(1))
  await vi.advanceTimersByTimeAsync(TICK_MS * 10)
  expect(refresh).toHaveBeenCalledTimes(1)
  release()
})

it('only counts down a holding row: a redirected or due one has no digits', () => {
  const countdown = new ReviewCountdown(vi.fn(), new Ticker())
  countdown.read(held(30, 'redirected'))
  expect(countdown.left('a')).toBeNull()
  countdown.read(held(0, 'due'))
  expect(countdown.left('a')).toBeNull()
})

it('ticks once a second while held, and stops when the last holder lets go', async () => {
  const clock = new Ticker()
  const first = clock.hold()
  const second = clock.hold()
  expect(clock.running).toBe(true)
  const start = clock.now
  await vi.advanceTimersByTimeAsync(TICK_MS * 2)
  expect(clock.now - start).toBe(TICK_MS * 2)
  first()
  first() // a second release of the same hold counts once
  expect(clock.running).toBe(true)
  second()
  expect(clock.running).toBe(false)
  const stopped = clock.now
  await vi.advanceTimersByTimeAsync(TICK_MS * 5)
  expect(clock.now).toBe(stopped)
})

it('stops while the tab is hidden and catches up when it is shown', async () => {
  const refresh = vi.fn()
  const clock = new Ticker()
  const countdown = new ReviewCountdown(refresh, clock)
  countdown.read(held(10))
  const release = countdown.hold()
  hidden = true
  document.dispatchEvent(new Event('visibilitychange'))
  expect(clock.running).toBe(false)
  await vi.advanceTimersByTimeAsync(TICK_MS * 20)
  expect(refresh).not.toHaveBeenCalled()
  hidden = false
  document.dispatchEvent(new Event('visibilitychange'))
  expect(clock.running).toBe(true)
  // shown again past the deadline: the countdown is at zero and asks once
  expect(countdown.left('a')).toBe(0)
  expect(refresh).toHaveBeenCalledTimes(1)
  release()
})

it('does not start while the tab is hidden', () => {
  hidden = true
  const clock = new Ticker()
  const release = clock.hold()
  expect(clock.running).toBe(false)
  release()
})
