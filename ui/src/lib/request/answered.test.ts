import { expect, it } from 'vitest'
import { answeredNow, answeredText, noteItem } from './answered'
import { entry, item, request } from './fixtures'
import type { LedgerNote } from '../types'

const at = (minute: number) =>
  `2026-01-01T00:${String(minute).padStart(2, '0')}:00+00:00`
const note = (id: string, minute: number): LedgerNote => ({
  id,
  text: `note ${id}`,
  created: at(minute),
  at: at(minute),
})

const before = request(
  [
    item('a', { notes: [note('n2', 5), note('n3', 2)] }),
    item('b', { notes: [note('n4', 8)] }),
  ],
  { notes: [note('n1', 5)], agent_wrote_at: at(3) },
)

it('names exactly the notes that went from waiting to answered', () => {
  // n3 was answered already; n4 was saved after the agent's new write
  const after = { ...before, agent_wrote_at: at(6) }
  expect(answeredNow(before, after).map((n) => n.id)).toEqual(['n1', 'n2'])
  expect(answeredNow(after, after)).toEqual([])
  expect(answeredNow(before, before)).toEqual([])
})

it('an edited or deleted note is not one the agent answered', () => {
  const after = request(
    [
      item('a', { notes: [{ ...note('n2', 5), at: at(9) }, note('n3', 2)] }),
      item('b', { notes: [note('n4', 8)] }),
    ],
    { notes: [], agent_wrote_at: at(6) },
  )
  expect(answeredNow(before, after)).toEqual([])
})

it('says what the agent last wrote, and where a note lives', () => {
  const after = request([], {
    title: 'Launch posters',
    history: [
      entry(4, 'agent', 'opened'),
      entry(6, 'agent', 'retook shot 3 warmer'),
      entry(7, 'user', 'note added'),
    ],
  })
  expect(answeredText(after)).toBe(
    'The agent updated Launch posters after your note: “retook shot 3 warmer”',
  )
  expect(answeredText({ ...after, history: [] })).toBe(
    'The agent updated Launch posters after your note.',
  )
  expect(noteItem(before, 'n4')).toBe('b')
  expect(noteItem(before, 'n1')).toBeNull()
})
