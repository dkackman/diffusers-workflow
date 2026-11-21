import { expect, it } from 'vitest'
import { formatMtime } from '../format'
import { noteLine, noteWaiting, notesWaiting } from './noteState'
import { entry, item, request } from './fixtures'
import type { LedgerRequest } from '../types'
import cases from '../../../../tests/fixtures/note_waiting_cases.json'

const at = (minute: number) =>
  `2026-01-01T00:${String(minute).padStart(2, '0')}:00+00:00`
const shown = (minute: number) => formatMtime(Date.parse(at(minute)) / 1000)
const note = { id: 'n1', text: 'warmer, please', created: at(5), at: at(5) }

it('a note with no agent write after it says nothing was sent', () => {
  const req = request([], {
    agent_wrote_at: at(1),
    history: [entry(1, 'agent', 'opened'), entry(6, 'user', 'comment on a')],
  })
  expect(noteLine(note, req)).toEqual({
    text: `Saved ${shown(5)}. Nothing is sent to the agent. It reads this the next time it updates the plan.`,
    answered: false,
  })
  expect(noteWaiting(note, req)).toBe(true)
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
    `Saved ${shown(5)}. The agent has updated the plan 2 times since (latest r14, ${shown(14)}: “retook shot 3 warmer”).`,
  )
  expect(line.text).not.toMatch(/\bread\b/)
  expect(line.answered).toBe(true)
  expect(noteWaiting(note, req)).toBe(false)
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
