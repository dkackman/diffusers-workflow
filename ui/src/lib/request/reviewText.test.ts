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
  holdsStopped,
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
// A hold is counted once the item would otherwise ask to be run
const held = (items: LedgerItem[]) =>
  planState(request(items.map((i) => ({ ...i, ready: true })))).held
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

it('says a hold with the agent connected, plural and singular, with and without estimates', () => {
  const two = group([
    item('a', { review: review('holding', LATER), estimate_minutes: 3 }),
    item('b', { review: review('holding'), estimate_minutes: 1.6 }),
    item('c', { review: review('redirected'), estimate_minutes: 50 }),
  ])
  expect(two.rows).toHaveLength(2)
  // the soonest deadline is the one named
  expect(two.first.item.key).toBe('b')
  expect(bannerText(two, true, 125)).toBe(
    `2 items waiting for your OK (about 5 min by the agent's estimate). The agent will run them at ${at} if it's connected (in 02:05), unless you leave a note.`,
  )
  const one = group([item('a', { review: review('holding') })])
  expect(bannerText(one, true, 3725)).toBe(
    `1 item waiting for your OK. The agent will run it at ${at} if it's connected (in 1:02:05), unless you leave a note.`,
  )
  const partly = group([
    item('a', { review: review('holding'), estimate_minutes: 2 }),
    item('b', { review: review('holding') }),
  ])
  expect(bannerText(partly, true, 60)).toBe(
    `2 items waiting for your OK (about 2 min by the agent's estimate). The agent will run them at ${at} if it's connected (in 01:00), unless you leave a note.`,
  )
})

it('says a hold with the agent not connected, plural and singular', () => {
  const two = group([
    item('a', { review: review('holding'), estimate_minutes: 3 }),
    item('b', { review: review('holding') }),
  ])
  expect(bannerText(two, false, 125)).toBe(
    `2 items waiting for your OK. The agent isn't connected, so nothing runs at ${at} unless it reconnects. Run them here, or leave a note.`,
  )
  const one = group([item('a', { review: review('holding') })])
  expect(bannerText(one, false, 125)).toBe(
    `1 item waiting for your OK. The agent isn't connected, so nothing runs at ${at} unless it reconnects. Run it here, or leave a note.`,
  )
})

it("a redirected hold says the truth when the agent isn't connected", () => {
  const redirected = group([item('a', { review: review('redirected') })])
  expect(bannerText(redirected, true, 0)).toBe(
    "You left a note after these were held, so they won't start on their own. The agent will read it and may revise them; run them here if they're right as they are.",
  )
  expect(bannerText(redirected, false, 0)).toBe(
    "You left a note after these were held, so they won't start on their own, and the agent isn't connected. Run them here, or wait for the agent.",
  )
})

it('says a due hold with and without a session', () => {
  const due = group([item('a', { review: review('due', UNTIL, 0) })])
  expect(bannerText(due, true, 0)).toBe(
    'Review time is up: the agent should be starting it.',
  )
  expect(bannerText(due, false, 0)).toBe(
    "Review time passed while the agent wasn't connected: nothing ran.",
  )
  const dueTwo = group([
    item('a', { review: review('due', UNTIL, 0) }),
    item('b', { review: review('due', UNTIL, 0) }),
  ])
  expect(bannerText(dueTwo, true, 0)).toBe(
    'Review time is up: the agent should be starting them.',
  )
})

it('offers Run now for the ready rows of a hold, a redirected one, or a due one nobody listens for', () => {
  const holding = group([
    item('a', { review: review('holding'), ready: true }),
    item('b', { review: review('holding'), status: 'revised' }),
  ])
  expect(runNowKeys(holding, true)).toEqual(['a'])
  expect(runNowKeys(holding, false)).toEqual(['a'])
  const none = group([
    item('b', { review: review('holding'), status: 'revised' }),
  ])
  expect(runNowKeys(none, false)).toEqual([])
  const due = group([item('a', { review: review('due'), ready: true })])
  expect(runNowKeys(due, false)).toEqual(['a'])
  expect(runNowKeys(due, true)).toEqual([])
  const redirected = group([
    item('a', { review: review('redirected'), ready: true }),
  ])
  expect(runNowKeys(redirected, false)).toEqual(['a'])
})

it('a redirected hold still offers Run now', () => {
  const redirected = group([
    item('a', { review: review('redirected'), ready: true }),
  ])
  expect(runNowKeys(redirected, true)).toEqual(['a'])
})

it('holdsStopped counts holding and due, not redirected', () => {
  const live = (key: string, state: string | null, dropped = false) => ({
    key,
    dropped,
    review: state ? review(state as never) : null,
  })
  const req = {
    items: [
      live('a', 'holding'),
      live('b', 'due'),
      live('c', 'redirected'),
      live('d', null),
      live('e', 'holding', true),
    ],
  } as never
  expect(holdsStopped(req)).toBe(2)
  expect(holdsStopped(req, null)).toBe(2)
  // an item's note redirects only that item's hold (ledger_views._hold)
  expect(holdsStopped(req, 'a')).toBe(1)
  expect(holdsStopped(req, 'b')).toBe(1)
  expect(holdsStopped(req, 'c')).toBe(0)
  expect(holdsStopped(req, 'd')).toBe(0)
  expect(holdsStopped(req, 'e')).toBe(0)
})

it('the held chip says waiting for your OK', () => {
  expect(chipText(review('holding'), 125)).toBe('waiting for your OK · 02:05')
  expect(chipText(review('holding'), 3600)).toBe(
    'waiting for your OK · 1:00:00',
  )
  expect(chipText(review('redirected'), null)).toBe('on hold: note left')
  expect(chipText(review('due'), null)).toBe('review time up')
})

it('has one sentence for each state, for the live region', () => {
  const one = group([item('a', { review: review('holding') })])
  expect(holdNews(one)).toBe('1 item waiting for your OK.')
  const two = group([
    item('a', { review: review('holding') }),
    item('b', { review: review('holding') }),
  ])
  expect(holdNews(two)).toBe('2 items waiting for your OK.')
  expect(holdNews(group([item('a', { review: review('redirected') })]))).toBe(
    "The held items won't start on their own: you left a note.",
  )
  expect(holdNews(group([item('a', { review: review('due') })]))).toBe(
    'Review time is up for the held items.',
  )
})
