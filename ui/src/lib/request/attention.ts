// A request as one row of a list: the line that says what it wants, the
// order the list puts it in, and its counts. All from a LedgerSummary.
import { bandsOf, type Band } from './itemState'
import { needsLine } from './standing'
import type { NeedsReason } from './itemState'
import type { LedgerItemStatus, LedgerSummary } from '../types'

export type AttentionKind =
  | 'needs'
  | 'failed'
  | 'running'
  | 'queued'
  | 'idle'
  | 'done-open'
  | 'empty'
  | 'closed'

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

/** What wants the user, by reason, as the server counted it; null from a
 * server older than the `needs` it sends. */
const needsOf = (summary: LedgerSummary) =>
  (summary.needs ?? null) as Record<NeedsReason, number> | null

/** Notes the agent has not read yet: the list's figure, never a need */
export const notesWaiting = (summary: LedgerSummary) =>
  summary.needs?.notes_waiting ?? 0

function openLine(summary: LedgerSummary): [AttentionKind, string] {
  const needs = needsOf(summary)
  const sentence = needs && needsLine(needs)
  if (sentence) return ['needs', sentence]
  const failed = of(summary, 'failed')
  if (!needs && failed) return ['failed', `${failed} failed`]
  if (of(summary, 'running')) return ['running', 'running']
  if (of(summary, 'queued')) return ['queued', 'queued']
  const planned = of(summary, 'planned') + of(summary, 'revised')
  if (planned) return ['idle', `${planned} planned, nothing running`]
  return liveItems(summary)
    ? ['done-open', 'all done, still open']
    : ['empty', 'nothing planned yet']
}

/** The row's one line, the first that holds: what needs the user (the
 * server's `needs`, said as the request header says it), else, from a server
 * that sends none, something failed; then running, queued, planned items
 * sitting idle, everything done and the request still open. A closed
 * request says nothing: its state tag does. */
export function attention(summary: LedgerSummary): Attention {
  const live = of(summary, 'running') > 0
  if (summary.state !== 'open') return { kind: 'closed', text: '', live }
  const [kind, text] = openLine(summary)
  return { kind, text, live }
}

/** An open request with something for the user to do: what the server's
 * `needs` counts (notes waiting are the agent's), else run, look at, or
 * keep a revised item's take */
export function needsYou(summary: LedgerSummary) {
  if (summary.state !== 'open') return false
  const needs = needsOf(summary)
  if (needs)
    return Object.entries(needs).some(
      ([reason, n]) => reason !== 'notes_waiting' && n > 0,
    )
  return (
    summary.ready > 0 || of(summary, 'failed') > 0 || of(summary, 'revised') > 0
  )
}

const inFlight = (summary: LedgerSummary) =>
  of(summary, 'running') + of(summary, 'queued') > 0

/** Held items first: they wait on the user and start on their own */
const rank = (summary: LedgerSummary) =>
  summary.state === 'open' && (needsOf(summary)?.held ?? 0) > 0
    ? 0
    : needsYou(summary)
      ? 1
      : inFlight(summary)
        ? 2
        : 3

/** The list's order: held requests, what else needs the user, then what is
 * in flight, then the most recently written. */
export function orderRequests(requests: LedgerSummary[]): LedgerSummary[] {
  const written = (s: LedgerSummary) =>
    s.updated_at ? Date.parse(s.updated_at) : 0
  return [...requests].sort(
    (a, b) => rank(a) - rank(b) || written(b) - written(a),
  )
}
