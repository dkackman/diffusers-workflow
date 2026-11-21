// Ledger records for the request tests: every field the server sends, so
// a test names only what it is about.
import type {
  LedgerAttempt,
  LedgerHistoryEntry,
  LedgerItem,
  LedgerRequest,
  LedgerSummary,
} from '../types'

export const attempt = (
  job: string,
  status: LedgerAttempt['status'] = 'succeeded',
  over: Partial<LedgerAttempt> = {},
): LedgerAttempt => ({ job, status, minutes: 1, output: null, ...over })

export const item = (
  key: string,
  over: Partial<LedgerItem> = {},
): LedgerItem => ({
  key,
  title: `Item ${key}`,
  depends_on: [],
  estimate_minutes: null,
  workflow: 'shot',
  arguments: {},
  dropped: false,
  accepted: null,
  comment: null,
  notes: [],
  status: 'planned',
  ready: false,
  attempts: [],
  result: null,
  spent_minutes: 0,
  ...over,
})

/** A done item: one succeeded job, which is its result */
export const doneItem = (
  key: string,
  over: Partial<LedgerItem> = {},
): LedgerItem =>
  item(key, {
    status: 'done',
    attempts: [attempt(`job-${key}`)],
    result: { job: `job-${key}`, run_id: `run-${key}`, output: null },
    spent_minutes: 1,
    ...over,
  })

export const entry = (
  revision: number,
  by: LedgerHistoryEntry['by'],
  why: string,
  over: Partial<LedgerHistoryEntry> = {},
): LedgerHistoryEntry => ({
  revision,
  by,
  why,
  at: `2026-01-01T00:${String(revision).padStart(2, '0')}:00+00:00`,
  ...over,
})

export const request = (
  items: LedgerItem[],
  over: Partial<LedgerRequest> = {},
): LedgerRequest => ({
  id: 'r1',
  title: 'Launch posters',
  state: 'open',
  revision: 3,
  prompt: null,
  goal: 'Three finished posters',
  comment: null,
  notes: [],
  agent_wrote_at: null,
  spent_minutes: 0,
  remaining_minutes: 0,
  items,
  history: [],
  ...over,
})

export const summary = (over: Partial<LedgerSummary> = {}): LedgerSummary => ({
  id: 'r1',
  title: 'Launch posters',
  state: 'open',
  revision: 2,
  items: 3,
  status_counts: { done: 2, planned: 1 },
  ready: 0,
  cover: [],
  spent_minutes: 4,
  remaining_minutes: 2,
  updated_at: null,
  ...over,
})
