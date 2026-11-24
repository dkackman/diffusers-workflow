// The shared cases of what "Needs you" counts: tests/test_ui_twins.py
// pins dw/ledger_views.needs_counts to the same file, so the list's needs
// and the request page's agree.
import { expect, it } from 'vitest'
import { planState } from './itemState'
import { notesWaiting } from './noteState'
import { attempt, item, request } from './fixtures'
import type { LedgerAttempt, LedgerItem, LedgerRequest } from '../types'
import shared from '../../../../tests/fixtures/request_needs_cases.json'

type SharedItem = Omit<Partial<LedgerItem>, 'attempts'> & {
  key: string
  attempts: { job: string; status: LedgerAttempt['status'] }[]
}
type SharedCase = {
  name: string
  request: Pick<LedgerRequest, 'state' | 'agent_wrote_at' | 'notes'> & {
    items: SharedItem[]
  }
  needs: Record<string, number>
  notes_waiting: number
}

const filled = (shape: SharedCase['request']): LedgerRequest =>
  request(
    shape.items.map(({ key, attempts, ...rest }) =>
      item(key, {
        ...rest,
        attempts: attempts.map((a) => attempt(a.job, a.status)),
      }),
    ),
    {
      state: shape.state,
      agent_wrote_at: shape.agent_wrote_at,
      notes: shape.notes,
    },
  )

it('every shared case gives its needs', () => {
  const cases = shared.cases as SharedCase[]
  expect(cases.map((c) => c.name)).toEqual(
    expect.arrayContaining([
      'held-ready',
      'failed-runnable',
      'failed-blocked',
      'failed-after-dropped',
      'cancelled',
      'revised-in-flight',
      'two-takes-unaccepted',
      'closed-request',
      'note-waiting',
    ]),
  )
  for (const c of cases) {
    const req = filled(c.request)
    expect({ name: c.name, needs: planState(req).needs }).toEqual({
      name: c.name,
      needs: c.needs,
    })
    expect({ name: c.name, waiting: notesWaiting(req) }).toEqual({
      name: c.name,
      waiting: c.notes_waiting,
    })
  }
})
