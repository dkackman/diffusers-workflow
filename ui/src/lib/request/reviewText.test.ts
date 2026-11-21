import { expect, it } from 'vitest'
import { item, request } from './fixtures'
import { planState } from './itemState'
import {
  bannerText,
  chipText,
  clockTime,
  countdown,
  holdGroup,
  holdNews,
  runNowKeys,
} from './reviewText'
import type { LedgerItem, LedgerReview } from '../types'

const UNTIL = '2026-11-21T15:30:00+00:00'
const LATER = '2026-11-21T16:00:00+00:00'
const review = (
  state: LedgerReview['state'],
  until = UNTIL,
  seconds_left = 125,
): LedgerReview => ({ until, state, seconds_left })
const held = (items: LedgerItem[]) => planState(request(items)).held
const group = (items: LedgerItem[]) => {
  const found = holdGroup(held(items))
  if (!found) throw new Error('no hold')
  return found
}
const at = clockTime(UNTIL)

it('counts down in MM:SS, and H:MM:SS past an hour', () => {
  expect(countdown(0)).toBe('00:00')
  expect(countdown(9)).toBe('00:09')
  expect(countdown(125)).toBe('02:05')
  expect(countdown(3599)).toBe('59:59')
  expect(countdown(3600)).toBe('1:00:00')
  expect(countdown(3600 * 23 + 61)).toBe('23:01:01')
  expect(countdown(-5)).toBe('00:00')
})

it("gives the deadline in the browser's locale time", () => {
  expect(at).toBe(
    new Date(UNTIL).toLocaleTimeString(undefined, {
      hour: '2-digit',
      minute: '2-digit',
    }),
  )
})

it('has no group when nothing is held', () => {
  expect(holdGroup(held([item('a')]))).toBeNull()
})

it('words the banner by the most urgent state: due, then holding, then redirected', () => {
  const all = [
    item('r', { review: review('redirected') }),
    item('h', { review: review('holding') }),
    item('d', { review: review('due') }),
  ]
  expect(group(all).state).toBe('due')
  expect(group(all.slice(0, 2)).state).toBe('holding')
  expect(group(all.slice(0, 1)).state).toBe('redirected')
})

it('says a hold with a session listening, plural and singular, with and without estimates', () => {
  const two = group([
    item('a', { review: review('holding', LATER), estimate_minutes: 3 }),
    item('b', { review: review('holding'), estimate_minutes: 1.6 }),
    item('c', { review: review('redirected'), estimate_minutes: 50 }),
  ])
  expect(two.rows).toHaveLength(2)
  // the soonest deadline is the one named
  expect(two.first.item.key).toBe('b')
  expect(bannerText(two, true, 125)).toBe(
    `2 items held for your review (about 5 min by the agent's estimate). A session is listening: they run at ${at} (in 02:05) unless you leave a note.`,
  )
  const one = group([item('a', { review: review('holding') })])
  expect(bannerText(one, true, 3725)).toBe(
    `1 item held for your review. A session is listening: it runs at ${at} (in 1:02:05) unless you leave a note.`,
  )
  const partly = group([
    item('a', { review: review('holding'), estimate_minutes: 2 }),
    item('b', { review: review('holding') }),
  ])
  expect(bannerText(partly, true, 60)).toBe(
    `2 items held for your review (about 2 min by the agent's estimate). A session is listening: they run at ${at} (in 01:00) unless you leave a note.`,
  )
})

it('says a hold with no session listening, plural and singular', () => {
  const two = group([
    item('a', { review: review('holding'), estimate_minutes: 3 }),
    item('b', { review: review('holding') }),
  ])
  expect(bannerText(two, false, 125)).toBe(
    `2 items held for your review. No session is listening, so nothing will run at ${at}. Run them here, or leave a note.`,
  )
  const one = group([item('a', { review: review('holding') })])
  expect(bannerText(one, false, 125)).toBe(
    `1 item held for your review. No session is listening, so nothing will run at ${at}. Run it here, or leave a note.`,
  )
})

it('says a redirected hold, and a due one with and without a session', () => {
  const redirected = group([item('a', { review: review('redirected') })])
  for (const listening of [true, false])
    expect(bannerText(redirected, listening, 0)).toBe(
      'You left a note after these were held: the agent will revise before running.',
    )
  const due = group([item('a', { review: review('due', UNTIL, 0) })])
  expect(bannerText(due, true, 0)).toBe(
    'Review time is up: the agent should be starting these.',
  )
  expect(bannerText(due, false, 0)).toBe(
    'Review time passed with no session listening: nothing ran.',
  )
})

it('offers Run now for the ready rows of a hold, or of a due one nobody listens for', () => {
  const holding = group([
    item('a', { review: review('holding'), ready: true }),
    item('b', { review: review('holding') }),
  ])
  expect(runNowKeys(holding, true)).toEqual(['a'])
  expect(runNowKeys(holding, false)).toEqual(['a'])
  const none = group([item('b', { review: review('holding') })])
  expect(runNowKeys(none, false)).toEqual([])
  const due = group([item('a', { review: review('due'), ready: true })])
  expect(runNowKeys(due, false)).toEqual(['a'])
  expect(runNowKeys(due, true)).toEqual([])
  const redirected = group([
    item('a', { review: review('redirected'), ready: true }),
  ])
  expect(runNowKeys(redirected, false)).toEqual([])
})

it("words a held row's chip", () => {
  expect(chipText(review('holding'), 125)).toBe('held · 02:05')
  expect(chipText(review('holding'), 3600)).toBe('held · 1:00:00')
  expect(chipText(review('redirected'), null)).toBe('held · note left')
  expect(chipText(review('due'), null)).toBe('review time up')
})

it('has one sentence for each state, for the live region', () => {
  const one = group([item('a', { review: review('holding') })])
  expect(holdNews(one)).toBe('1 item held for your review.')
  const two = group([
    item('a', { review: review('holding') }),
    item('b', { review: review('holding') }),
  ])
  expect(holdNews(two)).toBe('2 items held for your review.')
  expect(holdNews(group([item('a', { review: review('redirected') })]))).toBe(
    'You left a note after these were held: the agent will revise before running.',
  )
  expect(holdNews(group([item('a', { review: review('due') })]))).toBe(
    'Review time is up for the held items.',
  )
})
