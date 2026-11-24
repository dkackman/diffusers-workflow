import { expect, it } from 'vitest'
import { formatMtime } from '../format'
import {
  answerOf,
  LISTENING_LINE,
  NOT_LISTENING_LINE,
  noteAnswered,
  noteListShows,
  noteLine,
  noteWaiting,
  notesWaiting,
} from './noteState'
import { entry, item, request } from './fixtures'
import type { LedgerRequest } from '../types'
import cases from '../../../../tests/fixtures/note_waiting_cases.json'

const at = (minute: number) =>
  `2026-01-01T00:${String(minute).padStart(2, '0')}:00+00:00`
const shown = (minute: number) => formatMtime(Date.parse(at(minute)) / 1000)
const note = { id: 'n1', text: 'warmer, please', created: at(5), at: at(5) }

it('a waiting note says whether the agent is connected', () => {
  const req = request([], {
    agent_wrote_at: at(1),
    history: [entry(1, 'agent', 'opened'), entry(6, 'user', 'note added')],
  })
  expect(NOT_LISTENING_LINE).toBe(
    "The agent isn't connected. It reads this the next time it updates the plan.",
  )
  expect(LISTENING_LINE).toBe(
    'The agent is connected; it will see this within about 10 seconds.',
  )
  expect(noteLine(note, req)).toEqual({
    text: `Saved ${shown(5)}. ${NOT_LISTENING_LINE}`,
    answered: false,
  })
  expect(noteLine(note, req, new Date(), false).text).toContain(
    NOT_LISTENING_LINE,
  )
  expect(noteLine(note, req, new Date(), true)).toEqual({
    text: `Saved ${shown(5)}. ${LISTENING_LINE}`,
    answered: false,
  })
  expect(noteWaiting(note, req)).toBe(true)
  expect(noteAnswered(note, req)).toBe(false)
})

it('an edited note shows when it was saved and when it was edited', () => {
  const edited = { ...note, at: at(7) }
  const req = request([], { agent_wrote_at: at(6) })
  expect(noteLine(edited, req).text).toBe(
    `Saved ${shown(5)} · edited ${shown(7)}. ${NOT_LISTENING_LINE}`,
  )
  // the edit is what waits: the agent wrote between the two
  expect(noteWaiting(edited, req)).toBe(true)
})

it('waits when the agent has never written', () => {
  expect(noteWaiting(note, request([], { agent_wrote_at: null }))).toBe(true)
})

it('says the agent has written since, never that it read the note', () => {
  const req = request([], {
    agent_wrote_at: at(14),
    history: [
      entry(4, 'agent', 'before the note'),
      entry(8, 'agent', 'added shot 7'),
      entry(14, 'agent', 'retook shot 3 warmer'),
      entry(15, 'user', 'comment on b'),
    ],
  })
  const line = noteLine(note, req)
  expect(line.text).toBe(
    `Saved ${shown(5)}. The agent has updated the plan 2 times since (latest r14, ${shown(14)}).`,
  )
  expect(line.text).not.toMatch(/\bread\b/)
  expect(line.answered).toBe(true)
  expect(noteWaiting(note, req)).toBe(false)
  expect(noteAnswered(note, req)).toBe(true)
  // listening or not, an answered note says what the agent did
  expect(noteLine(note, req, new Date(), true).text).toBe(line.text)
  expect(
    noteLine(
      note,
      request([], {
        agent_wrote_at: at(8),
        history: [entry(8, 'agent', 'x')],
      }),
    ).text,
  ).toContain('updated the plan once since')
})

it('without agent entries in history, says only when the agent last wrote', () => {
  const req = request([], { agent_wrote_at: at(9), history: [] })
  expect(noteLine(note, req)).toEqual({
    text: `Saved ${shown(5)}. The agent has updated the plan since (${shown(9)}).`,
    answered: true,
  })
  expect(noteWaiting(note, req)).toBe(false)
})

it('says a closed request may not be returned to, and does not count its notes', () => {
  const req = request([], { state: 'abandoned', history: [] })
  expect(noteLine(note, req).text).toBe(
    `Saved ${shown(5)}. This request is closed. The agent may not return to it.`,
  )
  expect(noteLine(note, req, new Date(), true).text).not.toContain('listening')
  expect(noteWaiting(note, req)).toBe(false)
  expect(noteWaiting(null, request([]))).toBe(false)
})

it("counts the request's own waiting note with its live items'", () => {
  const answered = { ...note, id: 'n2', created: at(2), at: at(2) }
  const req = request(
    [
      item('a', { notes: [note] }),
      item('b', { notes: [answered] }),
      item('c', { notes: [note], dropped: true, status: 'dropped' }),
      item('d'),
    ],
    { notes: [note], agent_wrote_at: at(3) },
  )
  expect(notesWaiting(req)).toBe(2)
})

// The notes channel (dw_mcp/ledger.py: note_waiting) announces a note by
// this rule; tests/test_ui_twins.py runs the same cases against it
it.each(cases)('shared case: $name', ({ request: stored, waiting }) => {
  const req = stored as unknown as LedgerRequest
  const ids = [
    ...req.notes,
    ...req.items.filter((i) => !i.dropped).flatMap((i) => i.notes),
  ]
    .filter((n) => noteWaiting(n, req))
    .map((n) => n.id)
  expect(ids).toEqual(waiting)
  expect(notesWaiting(req)).toBe(waiting.length)
})

it("the answer is the agent's first write after the note", () => {
  const req = request([], {
    agent_wrote_at: at(14),
    history: [
      entry(4, 'agent', 'before the note'),
      entry(6, 'user', 'a comment'),
      entry(8, 'agent', 'added shot 7'),
      entry(14, 'agent', 'retook shot 3 warmer'),
    ],
  })
  expect(answerOf(note, req)).toEqual({ revision: 8, why: 'added shot 7' })
})

it('no write after it, no answer', () => {
  const req = request([], {
    agent_wrote_at: at(4),
    history: [entry(4, 'agent', 'before the note'), entry(9, 'user', 'x')],
  })
  expect(answerOf(note, req)).toBeNull()
})

it('a dropped item with no notes shows no list', () => {
  const req = request([item('c', { dropped: true, status: 'dropped' })])
  expect(noteListShows(req, 'c', 0)).toBe(false)
  expect(noteListShows(request([item('c')]), 'c', 0)).toBe(true)
})

it('a dropped item keeps its notes to read', () => {
  const req = request([
    item('c', { dropped: true, status: 'dropped', notes: [note] }),
  ])
  expect(noteListShows(req, 'c', 1)).toBe(true)
})
