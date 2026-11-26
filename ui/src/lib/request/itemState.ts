// What the request page says about each item of a plan. The server derives
// an item's `status` and `ready` from its linked jobs; everything else here
// is read off one LedgerRequest: what a planned item waits on, whether a
// failure was a cancel, which items want the user, which take is the result.
import { formatMinutes, formatSpent } from '../format'
import { PROMPT } from '../references'
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
  | 'revised'
  | 'failed'
  | 'cancelled'
  | 'dropped'

/** Why an item wants the user, only ever for what the user can act on: it
 * waits for their OK (held, in place of ready or revised), it can be run, its
 * result was made for a plan or prompt that has changed since, it failed and
 * can run again, it has two or more good takes and none accepted, or its
 * accepted take's files are gone */
export type NeedsReason =
  'held' | 'ready' | 'revised' | 'failed' | 'choose' | 'accepted_missing'

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
  /** "Run what can run now" would queue it: ready, or revised and runnable */
  runsNow: boolean
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
  /** How many live items "Run what can run now" would queue (`runsNow`) */
  toRun: number
  /** The live items held for the user's review that need them (`needs`
   * says 'held'), in plan order */
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
  // has a result but its plan changed since: outlined like ready, it waits on the user
  revised: { word: 'revised', tone: 'ready' },
  failed: { word: 'failed', tone: 'failed' },
  // grey, not .chip.cancelled: an item's chip is kept apart from a job's
  // status (ARCHITECTURE, UI status and grouping)
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

const inFlight = (item: LedgerItem) =>
  [...item.attempts]
    .reverse()
    .find((a) => a.status === 'running' || a.status === 'queued')

export function displayOf(item: LedgerItem, depsDone: boolean): ItemDisplay {
  if (item.status === 'failed') return failedDisplay(item)
  // The server keeps a revised item revised while its new take is out: the
  // page shows the take in flight, so it is not offered a second time
  if (item.status === 'revised') {
    const active = inFlight(item)
    if (active) return active.status as 'queued' | 'running'
  }
  if (item.status !== 'planned') return item.status
  if (item.ready) return 'ready'
  return depsDone ? 'planned' : 'waiting'
}

/** The displays a review hold stands in for: what would otherwise ask */
const ASKS: Partial<Record<ItemDisplay, NeedsReason>> = {
  ready: 'ready',
  revised: 'revised',
}

/** The same rule as `needs_counts` in dw/ledger_views.py, pinned by
 * tests/fixtures/request_needs_cases.json */
function needsOf(
  item: LedgerItem,
  display: ItemDisplay,
  open: boolean,
  runnable: boolean,
  held: boolean,
): NeedsReason[] {
  if (!open || item.dropped) return []
  const ask = ASKS[display]
  const needs: (NeedsReason | false | undefined)[] = [
    ask && (held ? 'held' : ask),
    display === 'failed' && runnable && 'failed',
    unchosen(item) && 'choose',
    item.accepted_missing && 'accepted_missing',
  ]
  return needs.filter((reason): reason is NeedsReason => !!reason)
}

function runnableNow(
  item: LedgerItem,
  display: ItemDisplay,
  depsDone: boolean,
  open: boolean,
): boolean {
  if (display === 'ready') return true
  const again =
    display === 'failed' || display === 'cancelled' || display === 'revised'
  return again && open && depsDone && !!item.workflow
}

const zero = <K extends string>(keys: readonly K[]) =>
  Object.fromEntries(keys.map((k) => [k, 0])) as Record<K, number>
const DISPLAYS = Object.keys(CHIP) as ItemDisplay[]
const REASONS = [
  'held',
  'ready',
  'failed',
  'revised',
  'choose',
  'accepted_missing',
] as const

function tally(live: ItemState[], dropped: number) {
  const counts = zero(DISPLAYS)
  const needs = zero(REASONS)
  let toRun = 0
  for (const state of live) {
    counts[state.display]++
    for (const reason of state.needs) needs[reason]++
    if (state.runsNow) toRun++
  }
  counts.dropped = dropped
  return { counts, needs, toRun }
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
    const runnable = runnableNow(item, display, depsDone(item), open)
    return {
      item,
      display,
      position: item.dropped ? 0 : ++place,
      after,
      blockers: after.filter((dep) => dep.display !== 'done'),
      neededBy: request.items
        .filter((o) => !o.dropped && o.depends_on.includes(item.key))
        .map((o) => ref(o.key)),
      noWorkflow: !item.workflow && display !== 'done' && display !== 'revised',
      runnable,
      canRetake: display === 'done' && open && !!item.workflow,
      runsNow: display === 'ready' || (display === 'revised' && runnable),
      needs: needsOf(item, display, open, runnable, !!item.review),
      review: (!item.dropped && item.review) || null,
    }
  })
  const live = items.filter((state) => !state.item.dropped)
  const dropped = items.filter((state) => state.item.dropped)
  const held = live.filter((state) => state.needs.includes('held'))
  return { items, live, dropped, held, ...tally(live, dropped.length) }
}

/** Whether a row starts expanded: something is happening to it, it went
 * wrong, its plan changed since its take, its takes want the user (none
 * chosen, or the chosen one's files gone), or it holds a note the agent has
 * not had the chance to read. The user's own toggle wins over this
 * (RequestItem). */
const OPENS: ItemDisplay[] = ['running', 'failed', 'revised']
export function opensByDefault(state: ItemState, noteWaiting: boolean) {
  if (state.item.dropped) return false
  if (OPENS.includes(state.display)) return true
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
  if (state.display === 'done' || state.display === 'revised')
    return n > 1 ? count(n, 'take', 'takes') : null
  if (state.display === 'failed' || state.display === 'cancelled')
    return `${count(n, 'take', 'takes')}, none succeeded`
  return null
}

/** A failed row's line while it cannot run again: what it waits on. A
 * dependency that can never be done (dropped, or gone from the plan) says
 * so. Null for any other row, for one with no stored workflow (only the
 * agent can run it), and on a closed request, where nothing runs again. */
export function blockedLine(state: ItemState, open = true): string | null {
  if (!open || state.noWorkflow) return null
  if (state.display !== 'failed' || state.runnable) return null
  if (!state.blockers.length) return null
  const titles = state.blockers.map((dep) =>
    dep.display === 'dropped' ? `${dep.title} (dropped)` : dep.title,
  )
  const verb = titles.length === 1 ? 'is' : 'are'
  return `Failed. It can run again once ${listed(titles)} ${verb} done.`
}

/** A failed or cancelled row's why: its newest failed or cancelled take
 * that has a reason, as "Take 2 failed: <reason>". Null when none has one
 * (a server older than the reasons, or a take the engine gave no error). */
export function failedLine(item: LedgerItem): string | null {
  const take = takesOf(item)
    .reverse()
    .find(
      (t) =>
        (t.attempt.status === 'failed' || t.attempt.status === 'cancelled') &&
        t.attempt.reason,
    )
  if (!take) return null
  const verb = take.attempt.status === 'failed' ? 'failed' : 'was cancelled'
  return `Take ${take.n} ${verb}: ${take.attempt.reason}`
}

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

/** The number the next take of an item gets: takes count from 1 */
export const nextTake = (item: LedgerItem) => item.attempts.length + 1

/** The take a revised row shows as its result, the one "Keep this take"
 * accepts again: null when its result has no take among its attempts */
export function shownTake(item: LedgerItem): Take | null {
  const job = item.result?.job
  return takesOf(item).find((take) => take.attempt.job === job) ?? null
}

const MOVED: Record<string, string> = {
  workflow: 'its workflow',
  arguments: 'its arguments',
}

/** One server reason in words: `prompt:<name>` is that saved prompt */
function moved(reason: string): string {
  if (reason.startsWith(PROMPT))
    return `the prompt ${reason.slice(PROMPT.length)}`
  return MOVED[reason] ?? reason
}

export const listed = (parts: string[]) =>
  parts.length <= 1
    ? (parts[0] ?? '')
    : `${parts.slice(0, -1).join(', ')} and ${parts.at(-1)}`

/** What a revised item's `revised_because` says, as a sentence: "The
 * prompt board-01 changed since take 1." Null for an item not revised. */
export function revisedLine(
  item: LedgerItem,
  take: Take | null,
): string | null {
  if (item.status !== 'revised') return null
  const parts = (item.revised_because ?? []).map(moved)
  const what = parts.length ? listed(parts) : 'its plan'
  const subject = what.charAt(0).toUpperCase() + what.slice(1)
  const since = take ? `take ${take.n}` : 'its result was made'
  return `${subject} changed since ${since}.`
}

/** The DOM id of a row's title button: what a dependency name, a progress
 * segment or an announcement's follow-up focuses. */
export const titleId = (key: string) => `req-item-${key}-title`
/** The DOM id of a row's open half (ItemMore), which its title controls */
export const moreId = (key: string) => `req-item-${key}-more`

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
