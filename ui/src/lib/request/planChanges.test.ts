import { expect, it } from 'vitest'
import { foldChanges, stateChange, who } from './planChanges'
import { entry } from './fixtures'

it("lists the history newest first, folding a run of the user's plain writes", () => {
  const changes = foldChanges([
    entry(1, 'agent', 'opened', { state: 'open', goal: 'Two posters' }),
    entry(2, 'user', 'comment on a'),
    entry(3, 'user', 'comment on b'),
    entry(4, 'user', 'comment on a'),
    entry(5, 'agent', 'retook shot 3 warmer'),
    entry(6, 'user', 'comment on c'),
  ])
  expect(
    changes.map((c) =>
      c.kind === 'entry' ? c.entry.revision : c.entries.map((e) => e.revision),
    ),
  ).toEqual([6, 5, [4, 3, 2], 1])
})

it('does not fold a user write that changed the goal or the state', () => {
  const changes = foldChanges([
    entry(1, 'user', 'comment on a'),
    entry(2, 'user', 'not needed', { state: 'abandoned' }),
    entry(3, 'user', 'comment on a'),
  ])
  expect(changes.map((c) => c.kind)).toEqual(['entry', 'entry', 'entry'])
})

it('says what a state on an entry means', () => {
  expect(
    stateChange(entry(1, 'agent', 'opened', { state: 'open' }), true),
  ).toBeNull()
  expect(stateChange(entry(5, 'user', 'again', { state: 'open' }), false)).toBe(
    'reopened',
  )
  expect(
    stateChange(entry(4, 'agent', 'all in', { state: 'done' }), false),
  ).toBe('marked done')
  expect(
    stateChange(entry(4, 'user', 'no', { state: 'abandoned' }), false),
  ).toBe('abandoned')
  expect(stateChange(entry(2, 'agent', 'x'), false)).toBeNull()
  expect(who(entry(2, 'user', 'x'))).toBe('you')
  expect(who(entry(2, 'agent', 'x'))).toBe('agent')
})
