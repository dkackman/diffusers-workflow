import { expect, it } from 'vitest'
import { announce } from './announce'
import { attempt, doneItem, item, request } from './fixtures'

const before = request([
  item('shot', {
    title: 'Shot 3',
    status: 'running',
    attempts: [attempt('j', 'running')],
  }),
  item('cut', { title: 'The cut', depends_on: ['shot'] }),
])

it('says nothing on the first read, or about another request', () => {
  expect(announce(null, before)).toBe('')
  expect(announce(before, { ...before, id: 'other' })).toBe('')
})

it('says nothing when a refresh changed nothing to act on', () => {
  expect(announce(before, { ...before, revision: 9 })).toBe('')
})

it('says what finished and what that made ready, in plan order', () => {
  const after = request([
    doneItem('shot', { title: 'Shot 3' }),
    item('cut', { title: 'The cut', depends_on: ['shot'], ready: true }),
  ])
  expect(announce(before, after)).toBe(
    'Shot 3 finished. The cut is now ready to run.',
  )
})

it('says a failure, a new item, a drop and a new take', () => {
  const failed = request([
    item('shot', {
      title: 'Shot 3',
      status: 'failed',
      attempts: [attempt('j', 'failed')],
    }),
    item('cut', {
      title: 'The cut',
      depends_on: ['shot'],
      dropped: true,
      status: 'dropped',
    }),
    item('new', { title: 'Shot 4' }),
  ])
  expect(announce(before, failed)).toBe(
    'Shot 3 failed. The cut was dropped from the plan. The agent added Shot 4.',
  )
  const one = request([doneItem('a', { title: 'Poster' })])
  const two = request([
    doneItem('a', {
      title: 'Poster',
      attempts: [attempt('j1'), attempt('j2')],
    }),
  ])
  expect(announce(one, two)).toBe('Poster has a new take.')
})

it('says the request was closed or reopened', () => {
  expect(announce(before, { ...before, state: 'done' })).toMatch(
    /^The request was marked done\./,
  )
  expect(announce({ ...before, state: 'done' }, before)).toMatch(
    /^The request was reopened\./,
  )
})

it('counts the changes past three instead of reading them all out', () => {
  const keys = ['a', 'b', 'c', 'd', 'e']
  const planned = request(keys.map((k) => item(k, { title: k.toUpperCase() })))
  const ready = request(
    keys.map((k) => item(k, { title: k.toUpperCase(), ready: true })),
  )
  expect(announce(planned, ready)).toBe(
    'A is now ready to run. B is now ready to run. C is now ready to run. And 2 more changes.',
  )
})

it('says the review banner changed state once, and nothing as it ticks', () => {
  const held = (state: 'holding' | 'redirected' | 'due', seconds_left = 60) =>
    request([
      item('a', {
        title: 'Title card',
        review: { until: '2026-11-21T12:00:00+00:00', state, seconds_left },
      }),
    ])
  const free = request([item('a', { title: 'Title card' })])
  expect(announce(free, held('holding'))).toBe('1 item held for your review.')
  // the same state with fewer seconds left: nothing to say
  expect(announce(held('holding', 60), held('holding', 30))).toBe('')
  expect(announce(held('holding'), held('redirected'))).toBe(
    'You left a note after these were held: the agent will revise before running.',
  )
  expect(announce(held('holding'), held('due', 0))).toBe(
    'Review time is up for the held items.',
  )
  expect(announce(held('due', 0), free)).toBe('')
})

it('announces a rename', () => {
  const renamed = request([
    item('shot', {
      title: 'Shot 3: the pier',
      status: 'running',
      attempts: [attempt('j', 'running')],
    }),
    item('cut', { title: 'The cut', depends_on: ['shot'] }),
  ])
  expect(announce(before, renamed)).toBe(
    'The agent renamed Shot 3 to Shot 3: the pier.',
  )
  // Renamed and finished in one refresh: both, the new name in the second
  const done = request([
    doneItem('shot', { title: 'Shot 3: the pier' }),
    item('cut', { title: 'The cut', depends_on: ['shot'] }),
  ])
  expect(announce(before, done)).toBe(
    'The agent renamed Shot 3 to Shot 3: the pier. Shot 3: the pier finished.',
  )
})
