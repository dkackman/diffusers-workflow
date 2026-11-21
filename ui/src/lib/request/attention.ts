// A request as one row of a list: the line that says what it wants, the
// order the list puts it in, and its counts. All from a LedgerSummary.
import { bandsOf, type Band } from './itemState'
import type { LedgerItemStatus, LedgerSummary } from '../types'

export type AttentionKind =
  'failed' | 'running' | 'queued' | 'idle' | 'done-open' | 'empty' | 'closed'

export type Attention = {
  kind: AttentionKind
  text: string
  /** One of its jobs is running: the row takes the running treatment */
  live: boolean
}

const of = (summary: LedgerSummary, status: LedgerItemStatus) =>
  summary.status_counts[status] ?? 0

/** Items still in the plan: the summary's `items` counts dropped ones too */
export const liveItems = (summary: LedgerSummary) =>
  Object.entries(summary.status_counts).reduce(
    (sum, [status, n]) => (status === 'dropped' ? sum : sum + n),
    0,
  )

/** "4/9": done of live */
export const doneFigure = (summary: LedgerSummary) =>
  `${of(summary, 'done')}/${liveItems(summary)}`

export const summaryBands = (summary: LedgerSummary): Band[] =>
  bandsOf(summary.status_counts)

function idle(summary: LedgerSummary): string {
  const planned = of(summary, 'planned')
  const ready = summary.ready ? ` (${summary.ready} ready)` : ''
  return `idle: ${planned} planned${ready}, nothing running`
}

function openLine(summary: LedgerSummary): [AttentionKind, string] {
  const failed = of(summary, 'failed')
  if (failed) return ['failed', `${failed} failed`]
  if (of(summary, 'running')) return ['running', 'running']
  if (of(summary, 'queued')) return ['queued', 'queued']
  if (of(summary, 'planned')) return ['idle', idle(summary)]
  return liveItems(summary)
    ? ['done-open', 'all done, still open']
    : ['empty', 'nothing planned yet']
}

/** The row's one line, the first that holds: something failed, something
 * is running, something is queued, planned items sit idle, everything is
 * done and the request is still open. A closed request says nothing: its
 * state tag does. */
export function attention(summary: LedgerSummary): Attention {
  const live = of(summary, 'running') > 0
  if (summary.state !== 'open') return { kind: 'closed', text: '', live }
  const [kind, text] = openLine(summary)
  return { kind, text, live }
}

/** An open request with something for the user to do: run or look at */
export const needsYou = (summary: LedgerSummary) =>
  summary.state === 'open' && (summary.ready > 0 || of(summary, 'failed') > 0)

const inFlight = (summary: LedgerSummary) =>
  of(summary, 'running') + of(summary, 'queued') > 0

const rank = (summary: LedgerSummary) =>
  needsYou(summary) ? 0 : inFlight(summary) ? 1 : 2

/** The list's order: what needs the user, then what is in flight, then
 * the most recently written. */
export function orderRequests(requests: LedgerSummary[]): LedgerSummary[] {
  const written = (s: LedgerSummary) =>
    s.updated_at ? Date.parse(s.updated_at) : 0
  return [...requests].sort(
    (a, b) => rank(a) - rank(b) || written(b) - written(a),
  )
}
