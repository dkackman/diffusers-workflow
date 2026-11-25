// What the run dialog offers before anything is priced: which items a run
// would queue, in plan order; which it would not and why; what becomes
// ready once those finish; what a plan's quote adds to a row; and why a
// row offered earlier no longer matches the plan as it is now.
import { formatMinutes } from '../format'
import { describeEstimate, describePlan, type PlanLine } from '../plan'
import type { LedgerReview, Plan, PlanEstimate } from '../types'
import { CHIP, takesOf, type ItemState, type PlanState } from './itemState'
import { clockTime } from './reviewText'

/** An item a job would be queued for, with what the job would run */
export type Candidate = {
  key: string
  title: string
  workflow: string
  arguments: Record<string, unknown>
  /** The item's jobs when it was offered: one beyond these is a run of it
   * made since */
  jobs: string[]
  /** The agent's own estimate for the item, to hold the quote against */
  planMinutes: number | null
  /** A failed item offered again: listed apart and unticked to start */
  optional?: boolean
}
/** An item left out, in words a row can show */
export type Skip = { key: string; title: string; reason: string }

export type RunPlan = {
  candidates: Candidate[]
  skipped: Skip[]
  /** One named item, not "what is ready": no checkboxes, and for a done
   * item a line on what the new take does to the result */
  single: boolean
  /** For another take of a done item: what stands as the result meanwhile */
  retake: string | null
}

const candidate = ({ item }: ItemState): Candidate => ({
  key: item.key,
  title: item.title,
  workflow: item.workflow as string,
  arguments: item.arguments ?? {},
  jobs: item.attempts.map((attempt) => attempt.job),
  planMinutes: item.estimate_minutes,
})

const waitsOn = (row: ItemState) =>
  `waits on ${row.blockers
    .map((dep) => `${dep.title} (${CHIP[dep.display].word})`)
    .join(', ')}`

/** Why "run what is ready" passes over a live item that is not done */
function skipReason(row: ItemState): string {
  if (row.display === 'revised' && !row.blockers.length && !row.item.workflow)
    return 'no stored workflow: ask the agent to run it'
  if (row.display === 'queued') return 'already queued'
  if (row.display === 'running') return 'already running'
  if (row.blockers.length) return waitsOn(row)
  if (row.noWorkflow) return 'no stored workflow: ask the agent to run it'
  return `${CHIP[row.display].word}: run it again from its row`
}

/** A failed item that can run again: `runnable` is the one rule the
 * needs line counts it by (itemState.ts) */
const retryable = (row: ItemState) => row.display === 'failed' && row.runnable

/** The run of everything that can run now, in plan order: ready items, and
 * revised ones whose dependencies are done; then the failed ones that can
 * run again, offered unticked. Done items are not listed: there is nothing
 * to say about them. */
export function readyPlan(plan: PlanState): RunPlan {
  const ready = plan.live.filter((row) => row.runsNow)
  const again = plan.live.filter(retryable)
  const rest = plan.live.filter(
    (row) => !row.runsNow && !retryable(row) && row.display !== 'done',
  )
  return {
    candidates: [
      ...ready.map(candidate),
      ...again.map((row) => ({ ...candidate(row), optional: true })),
    ],
    skipped: rest.map((row) => ({
      key: row.item.key,
      title: row.item.title,
      reason: skipReason(row),
    })),
    single: false,
    retake: null,
  }
}

/** The banner's "Run now…": the held rows named that are ready, in plan
 * order, ticked as a run of what is ready is */
export function heldPlan(plan: PlanState, keys: readonly string[]): RunPlan {
  const rows = plan.live.filter(
    (row) => keys.includes(row.item.key) && row.runsNow,
  )
  return {
    candidates: rows.map(candidate),
    skipped: [],
    single: false,
    retake: null,
  }
}

/** The line a held item's row carries in the dialog: running it is the
 * user's go-ahead, or, with the review over and the agent connected, the
 * agent may be starting it already */
export function heldLine(
  review: LedgerReview | null,
  listening: boolean,
): PlanLine | null {
  if (!review) return null
  if (review.state === 'due' && listening)
    return {
      text: 'Review time is up and the agent is connected: it may be starting this now.',
      tone: 'warn',
    }
  return {
    text: `Waiting for your OK until ${clockTime(review.until)}. Running it now is your go-ahead.`,
    tone: 'plain',
  }
}

/** What stands as a done or revised item's result while another take is
 * made */
export function retakeLine(row: ItemState): string | null {
  if (row.display !== 'done' && row.display !== 'revised') return null
  const accepted = takesOf(row.item).find((take) => take.mark === 'accepted')
  return accepted
    ? `Take ${accepted.n} stays the accepted result until you accept another.`
    : 'Nothing is accepted, so the new take becomes the result when it finishes.'
}

/** The run of one item: a ready one, a failed one again, or another take
 * of a done one. An item that cannot be run is the one skip. */
export function itemPlan(plan: PlanState, key: string): RunPlan {
  const row = plan.live.find((state) => state.item.key === key)
  const empty = { candidates: [], skipped: [], single: true, retake: null }
  if (!row) return empty
  if (row.runnable || row.canRetake)
    return { ...empty, candidates: [candidate(row)], retake: retakeLine(row) }
  const { title } = row.item
  return { ...empty, skipped: [{ key, title, reason: skipReason(row) }] }
}

/** A value as text whatever order its keys were written in */
const canonical = (value: unknown): string =>
  JSON.stringify(value, (_key, held: unknown) =>
    held && typeof held === 'object' && !Array.isArray(held)
      ? Object.fromEntries(Object.entries(held).sort())
      : held,
  )

/** Why a candidate offered earlier must not be queued from the plan as it
 * is now: gone or dropped, run since, rewritten, or no longer something
 * this page can run. Null while it still can. */
export function staleReason(
  offered: Candidate,
  plan: PlanState,
): string | null {
  const row = plan.items.find((state) => state.item.key === offered.key)
  if (!row) return 'It is no longer in the plan.'
  const { item } = row
  if (item.dropped) return 'It was dropped from the plan.'
  if (item.attempts.some((attempt) => !offered.jobs.includes(attempt.job)))
    return 'A job was queued for it after this opened.'
  const now = candidate(row)
  if (
    now.workflow !== offered.workflow ||
    canonical(now.arguments) !== canonical(offered.arguments)
  )
    return 'Its workflow or arguments changed after this opened. Close and open again to run it as it is now.'
  if (row.runnable || row.canRetake) return null
  return `It can no longer be run: ${skipReason(row)}.`
}

/** The items that become ready once `keys` have finished: waiting ones
 * with something stored to run, every unmet dependency among `keys`. */
export function becomesReady(
  plan: PlanState,
  keys: readonly string[],
): { key: string; title: string }[] {
  const running = new Set(keys)
  if (!running.size) return []
  return plan.live
    .filter(
      (row) =>
        row.display === 'waiting' &&
        !row.noWorkflow &&
        row.blockers.every((dep) => running.has(dep.key)),
    )
    .map((row) => ({ key: row.item.key, title: row.item.title }))
}

const names = (titles: string[]) => titles.join(', ')

/** The dialog's line on what follows: said so nobody expects the page to
 * start the next wave on its own */
export function afterwardsLine(
  plan: PlanState,
  keys: readonly string[],
): string | null {
  const next = becomesReady(plan, keys)
  if (!next.length) return null
  const these = keys.length === 1 ? 'this finishes' : 'these finish'
  const one = next.length === 1
  const titles = names(next.map((n) => n.title))
  return `When ${these}, ${next.length} more ${one ? 'becomes' : 'become'} ready: ${titles}. This page will not start ${one ? 'it' : 'them'} for you.`
}

/** A row's quote: the figure as `describeEstimate` reads it, or, with no
 * figure, a plain line and the engine's reason behind it */
export function quoteLine(estimate: PlanEstimate | null): {
  text: string
  tone: 'plain' | 'warn'
  detail?: string
} {
  const line = describeEstimate(estimate)
  if (estimate && estimate.basis !== 'unknown' && estimate.minutes !== null)
    return line
  const text = 'No time estimate for this workflow.'
  if (!estimate) return { text, tone: 'plain' }
  return {
    text,
    tone: 'plain',
    detail: estimate.unpriced_reason || line.text,
  }
}

/** A quote more than this far from the plan's own estimate is said */
const DRIFT = 0.25

/** "the plan said about 4 min" when the quote is not near what the agent
 * estimated for the item; null when it is, or either figure is missing */
export function planSaid(
  planMinutes: number | null,
  quoted: number | null | undefined,
): string | null {
  if (planMinutes == null || quoted == null || planMinutes <= 0) return null
  if (Math.abs(quoted - planMinutes) / planMinutes <= DRIFT) return null
  return `the plan said ${formatMinutes(planMinutes)}`
}

export const ALL_CACHED =
  'All steps cached: this would reproduce the last take, not make a new one. Ask the agent for a variation.'

/** Whether the step cache would serve every step of the run */
export const allCached = (plan: Plan) =>
  plan.cached_steps !== null &&
  plan.steps > 0 &&
  plan.cached_steps >= plan.steps

/** What a plan adds under a row's quote: what the cache would serve and
 * what must be downloaded first, in `describePlan`'s words - and, when
 * every step is cached, that the run would make nothing new. */
export function planNotes(plan: Plan): PlanLine[] {
  const lines = describePlan(plan).filter(
    (line) =>
      line.text.startsWith('needs download') || /\bcached\b/.test(line.text),
  )
  if (!allCached(plan)) return lines
  return [
    { text: ALL_CACHED, tone: 'warn' },
    ...lines.filter((line) => line.text.startsWith('needs download')),
  ]
}

/** A refused submit, in plain words, by the server's code */
export const RUN_REFUSALS: Record<string, string> = {
  request_closed: 'This request is closed. Reopen it to run its items.',
  item_dropped: 'This item was dropped from the plan.',
  unknown_item: 'This item is no longer in the plan.',
  unknown_request: 'This request no longer exists.',
}
