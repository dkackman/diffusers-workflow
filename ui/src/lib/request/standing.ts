// The request header's two sentences: where the ask stands, and what in it
// wants the user. Read off the page's item states (itemState.ts).
import { formatDay } from '../format'
import type { LedgerRequest } from '../types'
import type { ItemState, NeedsReason, PlanState } from './itemState'

export type Standing = { sentence: string; needs: string | null }

const count = (n: number, one: string, many: string) =>
  `${n} ${n === 1 ? one : many}`

function rendering(plan: PlanState): string {
  const running = plan.live.filter((state) => state.display === 'running')
  return running.length === 1
    ? `Rendering ${running[0].item.title}.`
    : `Rendering ${running.length} items.`
}

/** The item a waiting item is in the end waiting on: down the first unmet
 * dependency of each until one that is not itself waiting. */
function rootBlocker(plan: PlanState): ItemState | null {
  const byKey = new Map(plan.live.map((state) => [state.item.key, state]))
  const seen = new Set<string>()
  let at = plan.live.find((state) => state.display === 'waiting')
  while (at && at.display === 'waiting' && !seen.has(at.item.key)) {
    seen.add(at.item.key)
    at = byKey.get(at.blockers[0]?.key ?? '')
  }
  return at && at.display !== 'waiting' ? at : null
}

function whyStuck(root: ItemState): string {
  if (root.display === 'failed') return ', which failed'
  if (root.display === 'cancelled') return ', which was cancelled'
  if (root.display === 'planned' && root.noWorkflow)
    return ', which is planned with no workflow'
  return ''
}

function waiting(plan: PlanState): string {
  const n = plan.counts.waiting
  const subject = n === 1 ? '1 item waits' : `${n} items wait`
  const root = rootBlocker(plan)
  if (!root) return `${subject} on other items.`
  return `${subject} on ${root.item.title}${whyStuck(root)}.`
}

/** "Marked done by the agent on Nov 20.": the history entry that set the
 * state says who and when; a ledger older than that field gives its last. */
function closed(request: LedgerRequest, now: Date): string {
  const word = request.state === 'done' ? 'Marked done' : 'Abandoned'
  const history = [...request.history].sort((a, b) => b.revision - a.revision)
  const entry = history.find((e) => e.state === request.state) ?? history[0]
  if (!entry) return `${word}.`
  const who = entry.by === 'user' ? 'you' : 'the agent'
  return `${word} by ${who} on ${formatDay(Date.parse(entry.at) / 1000, now)}.`
}

function lead(request: LedgerRequest, plan: PlanState, now: Date): string {
  const counts = plan.counts
  if (counts.running) return rendering(plan)
  if (counts.queued) return `${counts.queued} queued behind other jobs.`
  if (request.state !== 'open') return closed(request, now)
  if (counts.ready) return `Nothing is running. ${counts.ready} ready to run.`
  if (counts.waiting) return waiting(plan)
  if (plan.live.length === 0) return 'Nothing is planned yet.'
  return counts.done === plan.live.length
    ? 'Everything is done. The agent has not closed this request.'
    : 'Nothing is running.'
}

const NEEDS: Record<NeedsReason, (n: number) => string> = {
  ready: (n) => `${n} ready to run`,
  failed: (n) => `${n} failed`,
  choose: (n) => count(n, 'take to choose', 'takes to choose'),
  accepted_missing: (n) =>
    count(n, 'accepted take missing', 'accepted takes missing'),
}

/** "Needs you: 2 ready to run, 1 failed.", or null when nothing does */
export function needsLine(needs: Record<NeedsReason, number>): string | null {
  const parts = (Object.keys(NEEDS) as NeedsReason[])
    .filter((reason) => needs[reason] > 0)
    .map((reason) => NEEDS[reason](needs[reason]))
  return parts.length ? `Needs you: ${parts.join(', ')}.` : null
}

/** Where the request stands, as the first of these that holds: something
 * is rendering, something is queued, the request is closed, something is
 * ready, everything left is waiting, everything is done. Then how much of
 * the live plan is done. */
export function standing(
  request: LedgerRequest,
  plan: PlanState,
  now: Date = new Date(),
): Standing {
  const live = plan.live.length
  const tail = live ? ` ${plan.counts.done} of ${live} done.` : ''
  return {
    sentence: lead(request, plan, now) + tail,
    needs: needsLine(plan.needs),
  }
}

export type AgentSaid = { revision: number; why: string; at: string }

/** What the agent last wrote on an open request: its newest history entry,
 * whose `why` is how it answers - "ready for take 2, not run", "awaiting
 * go-ahead". Null on a closed request, or one the agent never wrote */
export function agentSaid(request: LedgerRequest): AgentSaid | null {
  if (request.state !== 'open') return null
  const latest = request.history
    .filter((e) => e.by === 'agent')
    .reduce<(typeof request.history)[number] | null>(
      (best, e) => (!best || e.revision > best.revision ? e : best),
      null,
    )
  return latest
    ? { revision: latest.revision, why: latest.why, at: latest.at }
    : null
}
