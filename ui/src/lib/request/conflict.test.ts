import { expect, it } from 'vitest'
import { culprit, itemDiff } from './conflict'
import { attempt, entry, item } from './fixtures'

it('says what changed about an item on the fields a decision rests on', () => {
  const before = item('a', { attempts: [attempt('j1')] })
  expect(itemDiff(before, before)).toEqual([])
  expect(
    itemDiff(before, {
      ...before,
      attempts: [attempt('j1'), attempt('j2'), attempt('j3')],
    }),
  ).toEqual(['2 new takes'])
  expect(itemDiff(before, { ...before, title: 'Renamed' })).toEqual([
    'it is now called Renamed',
  ])
  expect(itemDiff(before, { ...before, workflow: 'other' })).toEqual([
    'what it runs changed',
  ])
  expect(itemDiff(before, { ...before, arguments: { seed: 2 } })).toEqual([
    'its arguments changed',
  ])
  expect(itemDiff(before, { ...before, depends_on: ['z'] })).toEqual([
    'what it waits on changed',
  ])
  expect(itemDiff(before, { ...before, dropped: true })).toEqual([
    'it was dropped',
  ])
})

it('does not count what a decision does not rest on', () => {
  const before = item('a')
  expect(
    itemDiff(before, {
      ...before,
      status: 'running',
      ready: true,
      comment: { text: 'n', at: 't' },
      accepted: { job: 'j', run_id: null },
      estimate_minutes: 9,
      spent_minutes: 3,
    }),
  ).toEqual([])
})

it("names the agent's latest write since, or failing that the latest of any", () => {
  const history = [
    entry(3, 'agent', 'opened'),
    entry(4, 'agent', 'added shot 7'),
    entry(5, 'user', 'comment on a'),
  ]
  expect(culprit(history, 3)?.why).toBe('added shot 7')
  expect(culprit(history, 4)?.why).toBe('comment on a')
  expect(culprit(history, 5)).toBeNull()
  // up to, not including, the page's own write
  expect(culprit(history, 3, 4)).toBeNull()
})
