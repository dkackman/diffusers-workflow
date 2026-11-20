import { expect, it } from 'vitest'
import { formatMtime } from '../format'
import { noteLine, noteWaiting, notesWaiting } from './noteState'
import { entry, item, request } from './fixtures'
import type { LedgerRequest } from '../types'
import cases from '../../../../tests/fixtures/note_waiting_cases.json'

const at = (minute: number) =>
  `2026-01-01T00:${String(minute).padStart(2, '0')}:00+00:00`
const shown = (minute: number) => formatMtime(Date.parse(at(minute)) / 1000)
const note = { text: 'warmer, please', at: at(5) }

it('a note with no agent write after it says nothing was sent', () => {
  const req = request([], {
    history: [entry(1, 'agent', 'opened'), entry(6, 'user', 'comment on a')],
  })
  expect(noteLine(note, req)).toEqual({
    text: `Saved ${shown(5)}. Nothing is sent to the agent. It reads this the next time it updates the plan.`,
    answered: false,
  })
  expect(noteWaiting(note, req)).toBe(true)
})

it('says the agent has written since, never that it read the note', () => {
  const req = request([], {
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
    noteLine(note, request([], { history: [entry(8, 'agent', 'x')] })).text,
  ).toContain('updated the plan once since')
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
  const req = request(
    [
      item('a', { comment: note }),
      item('b', { comment: { text: 'answered', at: at(2) } }),
      item('c', { comment: note, dropped: true, status: 'dropped' }),
      item('d'),
    ],
    { comment: note, history: [entry(3, 'agent', 'replanned')] },
  )
  expect(notesWaiting(req)).toBe(2)
})

// The notes channel (dw_mcp/ledger.py: note_waiting) announces a note by
// this rule; tests/test_ui_twins.py runs the same cases against it
it.each(cases)('shared case: $name', ({ request: stored, waiting }) => {
  const req = stored as unknown as LedgerRequest
  const keys = [
    ...(noteWaiting(req.comment, req) ? [''] : []),
    ...req.items
      .filter((i) => !i.dropped && noteWaiting(i.comment, req))
      .map((i) => i.key),
  ]
  expect(keys).toEqual(waiting)
  expect(notesWaiting(req)).toBe(waiting.length)
})
