// What the request page says about each item of a plan. The server derives
// an item's `status` and `ready` from its linked jobs; everything else here
// is read off one LedgerRequest: what a planned item waits on, whether a
// failure was a cancel, which items want the user, which take is the result.
import { formatMinutes, formatSpent } from '../format'
import type {
  LedgerAttempt,
  LedgerItem,
  LedgerRequest,
  LedgerReview,
} from '../types'

export type ItemDisplay =
  | 'ready'
  | 'waiting'
  | 'planned'
  | 'queued'
  | 'running'
  | 'done'
  | 'failed'
  | 'cancelled'
  | 'dropped'

/** Why an item wants the user: it can be run, it failed, it has two or
 * more good takes and none accepted, or its accepted take's files are gone */
export type NeedsReason = 'ready' | 'failed' | 'choose' | 'accepted_missing'

/** Another item as a row names it: a dependency, or one that depends on it */
export type ItemRef = { key: string; title: string; display: ItemDisplay }

export type ItemState = {
  item: LedgerItem
  display: ItemDisplay
  /** 1-based place among the live items; 0 for a dropped one */
  position: number
  /** Its dependencies, in the order the plan lists them */
  after: ItemRef[]
  /** The dependencies that are not done */
  blockers: ItemRef[]
  /** The live items that depend on it */
  neededBy: ItemRef[]
  /** Planned or failed with nothing stored to run: only the agent can run it */
  noWorkflow: boolean
  /** A job could be queued for it from this page now: ready, or failed or
   * cancelled with its dependencies done and a stored workflow */
  runnable: boolean
  /** Done, and another take could be queued (from its menu only) */
  canRetake: boolean
  needs: NeedsReason[]
  /** The server's review hold on it, while one applies; null otherwise.
   * Its `state` is the one shown: the page never derives its own */
  review: LedgerReview | null
}

export type PlanState = {
  items: ItemState[]
  live: ItemState[]
  dropped: ItemState[]
  counts: Record<ItemDisplay, number>
  needs: Record<NeedsReason, number>
  /** The live items held for the user's review, in plan order */
  held: ItemState[]
}

/** The chip's word, and the `.chip` class in app.css that colours it */
export const CHIP: Record<ItemDisplay, { word: string; tone: string }> = {
  ready: { word: 'ready', tone: 'ready' },
  waiting: { word: 'waiting', tone: 'idle' },
  planned: { word: 'planned', tone: 'idle' },
  queued: { word: 'queued', tone: 'queued' },
  running: { word: 'running', tone: 'running' },
  done: { word: 'done', tone: 'succeeded' },
  failed: { word: 'failed', tone: 'failed' },
  // not .chip.cancelled: its --warn text is the --live hex
  cancelled: { word: 'cancelled', tone: 'idle' },
  dropped: { word: 'dropped', tone: 'idle' },
}

const succeeded = (item: LedgerItem) =>
  item.attempts.filter((a) => a.status === 'succeeded').length

/** Two or more good takes and none accepted: the result is only "latest" */
export const unchosen = (item: LedgerItem) =>
  !item.accepted && succeeded(item) >= 2

/** The server's `failed` covers "every attempt was cancelled" too */
function failedDisplay(item: LedgerItem): ItemDisplay {
  const cancelled =
    item.attempts.length > 0 &&
    item.attempts.every((a) => a.status === 'cancelled')
  return cancelled ? 'cancelled' : 'failed'
}

export function displayOf(item: LedgerItem, depsDone: boolean): ItemDisplay {
  if (item.status === 'failed') return failedDisplay(item)
  if (item.status !== 'planned') return item.status
  if (item.ready) return 'ready'
  return depsDone ? 'planned' : 'waiting'
}

function needsOf(
  item: LedgerItem,
  display: ItemDisplay,
  open: boolean,
): NeedsReason[] {
  if (!open || item.dropped) return []
  const needs: NeedsReason[] = []
  if (display === 'ready') needs.push('ready')
  if (display === 'failed') needs.push('failed')
  if (unchosen(item)) needs.push('choose')
  if (item.accepted_missing) needs.push('accepted_missing')
  return needs
}

function runnableNow(
  item: LedgerItem,
  display: ItemDisplay,
  depsDone: boolean,
  open: boolean,
): boolean {
  if (display === 'ready') return true
  const retry = display === 'failed' || display === 'cancelled'
  return retry && open && depsDone && item.workflow !== null
}

const zero = <K extends string>(keys: readonly K[]) =>
  Object.fromEntries(keys.map((k) => [k, 0])) as Record<K, number>
const DISPLAYS = Object.keys(CHIP) as ItemDisplay[]
const REASONS = ['ready', 'failed', 'choose', 'accepted_missing'] as const

function tally(live: ItemState[], dropped: number) {
  const counts = zero(DISPLAYS)
  const needs = zero(REASONS)
  for (const state of live) {
    counts[state.display]++
    for (const reason of state.needs) needs[reason]++
  }
  counts.dropped = dropped
  return { counts, needs }
}

/** Every item of `request` as the page shows it, in plan order. */
export function planState(request: LedgerRequest): PlanState {
  const open = request.state === 'open'
  const byKey = new Map(request.items.map((item) => [item.key, item]))
  const depsDone = (item: LedgerItem) =>
    item.depends_on.every((key) => byKey.get(key)?.status === 'done')
  const displays = new Map(
    request.items.map((item) => [item.key, displayOf(item, depsDone(item))]),
  )
  // A key the plan no longer holds reads as dropped: it cannot be done
  const ref = (key: string): ItemRef => ({
    key,
    title: byKey.get(key)?.title ?? key,
    display: displays.get(key) ?? 'dropped',
  })
  let place = 0
  const items = request.items.map((item): ItemState => {
    const display = displays.get(item.key) ?? 'dropped'
    const after = item.depends_on.map(ref)
    return {
      item,
      display,
      position: item.dropped ? 0 : ++place,
      after,
      blockers: after.filter((dep) => dep.display !== 'done'),
      neededBy: request.items
        .filter((o) => !o.dropped && o.depends_on.includes(item.key))
        .map((o) => ref(o.key)),
      noWorkflow: item.workflow === null && display !== 'done',
      runnable: runnableNow(item, display, depsDone(item), open),
      canRetake: display === 'done' && open && item.workflow !== null,
      needs: needsOf(item, display, open),
      review: (!item.dropped && item.review) || null,
    }
  })
  const live = items.filter((state) => !state.item.dropped)
  const dropped = items.filter((state) => state.item.dropped)
  const held = live.filter((state) => state.review !== null)
  return { items, live, dropped, held, ...tally(live, dropped.length) }
}

/** Whether a row starts expanded: something is happening to it, it went
 * wrong, its takes want the user (none chosen, or the chosen one's files
 * gone), or it holds a note the agent has not had the chance to read. The
 * user's own toggle wins over this (RequestItem). */
export function opensByDefault(state: ItemState, noteWaiting: boolean) {
  if (state.item.dropped) return false
  if (state.display === 'running' || state.display === 'failed') return true
  const takes =
    state.needs.includes('choose') || state.needs.includes('accepted_missing')
  return takes || noteWaiting
}

/** A linked job as the takes strip shows it. `accepted` is the take the
 * user chose; `latest` the one standing as the result with none chosen (or
 * with the chosen one's files gone, which is then `accepted-gone`). */
export type Take = {
  attempt: LedgerAttempt
  n: number
  mark: 'accepted' | 'accepted-gone' | 'latest' | null
}

function markOf(item: LedgerItem, job: string): Take['mark'] {
  const chosen = item.accepted?.job === job
  if (chosen) return item.accepted_missing ? 'accepted-gone' : 'accepted'
  return item.result?.job === job ? 'latest' : null
}

/** The item's takes, oldest first, numbered from 1. */
export const takesOf = (item: LedgerItem): Take[] =>
  item.attempts.map((attempt, index) => ({
    attempt,
    n: index + 1,
    mark: markOf(item, attempt.job),
  }))

/** How the result came to be the result: null when there is none */
export function resultMark(item: LedgerItem): 'accepted' | 'latest' | null {
  if (!item.result) return null
  return item.accepted && !item.accepted_missing ? 'accepted' : 'latest'
}

const count = (n: number, one: string, many: string) =>
  `${n} ${n === 1 ? one : many}`

/** The collapsed row's line about its takes: how many there are for a done
 * item with more than one, and that none worked for a failed one. */
export function takesLine(state: ItemState): string | null {
  const n = state.item.attempts.length
  if (state.display === 'done') return n > 1 ? count(n, 'take', 'takes') : null
  if (state.display === 'failed' || state.display === 'cancelled')
    return `${count(n, 'attempt', 'attempts')}, none succeeded`
  return null
}

const inFlight = (item: LedgerItem) =>
  [...item.attempts]
    .reverse()
    .find((a) => a.status === 'running' || a.status === 'queued')

/** The job a queued or running row's "Open job" leads to */
export const activeJob = (item: LedgerItem): string | null =>
  inFlight(item)?.job ?? null

/** The row's one figure: the estimate before a run, the minutes so far
 * during one, the minutes spent after. Empty when there is none to give. */
export function figureOf(state: ItemState): string {
  const { item, display } = state
  if (display === 'running') {
    const minutes = inFlight(item)?.minutes
    return minutes == null ? '' : `${formatSpent(minutes)} so far`
  }
  const before = ['ready', 'waiting', 'planned', 'queued'].includes(display)
  if (before)
    return item.estimate_minutes == null
      ? ''
      : `est. ${formatMinutes(item.estimate_minutes)}`
  return item.spent_minutes > 0 ? formatSpent(item.spent_minutes) : ''
}

/** The DOM id of a row's title button: what a dependency name, a progress
 * segment or an announcement's follow-up focuses. */
export const titleId = (key: string) => `req-item-${key}-title`

/** A progress segment's colour class (RequestProgress) */
export type ProgressTone = 'done' | 'running' | 'failed' | 'queued' | 'rest'
export const toneOf = (display: string): ProgressTone =>
  display === 'done' ||
  display === 'running' ||
  display === 'failed' ||
  display === 'queued'
    ? display
    : 'rest'

export type Band = { tone: ProgressTone; count: number; label: string }
const BAND_ORDER: ProgressTone[] = [
  'done',
  'running',
  'failed',
  'queued',
  'rest',
]
const BAND_WORD: Record<ProgressTone, string> = {
  done: 'done',
  running: 'running',
  failed: 'failed',
  queued: 'queued',
  rest: 'not started',
}

/** Status-proportional bands for a plan too long for one segment an item,
 * or for a list row that only has counts: `counts` is by item status or
 * display state, dropped items left out. */
export function bandsOf(counts: Record<string, number>): Band[] {
  const totals = zero(BAND_ORDER)
  for (const [name, n] of Object.entries(counts))
    if (name !== 'dropped') totals[toneOf(name)] += n
  return BAND_ORDER.filter((tone) => totals[tone] > 0).map((tone) => ({
    tone,
    count: totals[tone],
    label: `${totals[tone]} ${BAND_WORD[tone]}`,
  }))
}
