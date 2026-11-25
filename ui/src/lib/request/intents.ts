// Every edit the request page makes is an intent - "accept this take",
// "drop this item", "put X before Y" - never an edited copy of the plan.
// An intent is applied to whatever the server last said, so the same edit
// can be made again over a plan the agent has rewritten since, or be
// refused by name when it no longer means anything.
import type { RequestWriteBody, WrittenItem } from '../apiRequest'
import type { LedgerItem, LedgerRequest } from '../types'
import { placeBefore } from './reorderMath'

/** What a drop remembers so that undoing it puts everything back */
export type DropMemory = {
  item: LedgerItem
  /** The live item it sat before; null when it was last */
  before: string | null
  /** The live items that depended on it */
  dependents: string[]
}

export type Intent =
  /** Choose a take as the item's result; a null job clears the choice */
  | { kind: 'accept'; key: string; job: string | null }
  | { kind: 'drop'; key: string }
  /** `memory` is what this page remembers of a drop it made: without it
   * the item goes after the last live one */
  | { kind: 'restore'; key: string; memory?: DropMemory }
  /** Put `key` before `before`, or last among the live items */
  | { kind: 'move'; key: string; before: string | null }
  /** `from` is the state the write expects to find: an Undo puts back
   * the state it took away only over the one it wrote */
  | {
      kind: 'state'
      state: LedgerRequest['state']
      from?: LedgerRequest['state']
    }

export type Refusal =
  /** The item is no longer in the plan at all */
  | 'item_gone'
  /** The item was dropped from the plan */
  | 'item_dropped'
  /** The take is not one of the item's succeeded jobs */
  | 'attempt_gone'
  /** The item a move was relative to is not a live item */
  | 'anchor_gone'
  /** The plan already says what the intent asks */
  | 'no_change'
  /** The request's state is no longer the one the intent was made over */
  | 'moved_on'

export type Refused = { refusal: Refusal }
/** The request as the intent leaves it, the `why` its write carries, and
 * the dependencies a restore had to leave behind */
export type Change = { request: LedgerRequest; why: string; stripped: string[] }
export type Applied = { body: RequestWriteBody; stripped: string[] }

const no = (refusal: Refusal): Refused => ({ refusal })
export const isRefused = (result: object): result is Refused =>
  'refusal' in result

/** The server takes one line of at most 200 characters */
const WHY_MAX = 200
const why = (text: string) => text.replace(/\s+/g, ' ').slice(0, WHY_MAX)

const find = (request: LedgerRequest, key: string) =>
  request.items.find((item) => item.key === key)

/** A planned item's `ready`, as the server derives it, after its
 * dependencies changed under an edit not yet written */
function settle(request: LedgerRequest, items: LedgerItem[]): LedgerItem[] {
  const done = new Set(
    items.filter((i) => !i.dropped && i.status === 'done').map((i) => i.key),
  )
  const open = request.state === 'open'
  return items.map((item) => {
    const waits = item.status === 'planned' || item.status === 'revised'
    if (item.dropped || !waits) return item
    const ready =
      open && !!item.workflow && item.depends_on.every((k) => done.has(k))
    return ready === item.ready ? item : { ...item, ready }
  })
}

/** Live items in their order, then the dropped ones: the order kept */
function withItems(request: LedgerRequest, items: LedgerItem[]): LedgerRequest {
  const live = items.filter((item) => !item.dropped)
  const dropped = items.filter((item) => item.dropped)
  return { ...request, items: settle(request, [...live, ...dropped]) }
}

const replace = (request: LedgerRequest, next: LedgerItem) =>
  withItems(
    request,
    request.items.map((item) => (item.key === next.key ? next : item)),
  )

/** The item's live target, or why there is none */
function liveItem(request: LedgerRequest, key: string): LedgerItem | Refused {
  const item = find(request, key)
  if (!item) return no('item_gone')
  return item.dropped ? no('item_dropped') : item
}

/** What stands as the result with `job` accepted, or with none (the
 * latest succeeded take) */
function resultFor(item: LedgerItem, job: string | null): LedgerItem['result'] {
  const good = item.attempts.filter((a) => a.status === 'succeeded')
  const attempt = job ? good.find((a) => a.job === job) : good.at(-1)
  if (!attempt) return null
  const run = item.result?.job === attempt.job ? item.result.run_id : null
  return { job: attempt.job, run_id: run, output: attempt.output }
}

function accept(
  request: LedgerRequest,
  { key, job }: { key: string; job: string | null },
): Change | Refused {
  const item = liveItem(request, key)
  if (isRefused(item)) return item
  // Accepting a revised item's take again is the "keep this take" write:
  // the server records the basis it was kept against and the item is done
  const keep = item.status === 'revised' && job !== null
  if (!keep && (item.accepted?.job ?? null) === job) return no('no_change')
  const taken = item.attempts.some(
    (a) => a.job === job && a.status === 'succeeded',
  )
  if (job !== null && !taken) return no('attempt_gone')
  const next: LedgerItem = {
    ...item,
    accepted: job === null ? null : { job, run_id: null },
    accepted_missing: false,
    result: resultFor(item, job),
    ...(keep
      ? { status: 'done' as const, revised_because: [], ready: false }
      : {}),
  }
  return {
    request: replace(request, next),
    why: why(
      job === null
        ? `clear the accepted take of ${key}`
        : keep
          ? `keep ${job} for ${key}`
          : `accept ${job} for ${key}`,
    ),
    stripped: [],
  }
}

/** Dropping an item others depend on takes it out of their `depends_on`
 * in the same write: the server refuses a dependency on a key that is not
 * live. */
function drop(request: LedgerRequest, key: string): Change | Refused {
  const item = find(request, key)
  if (!item) return no('item_gone')
  if (item.dropped) return no('no_change')
  const items = request.items.map((other): LedgerItem => {
    if (other.key === key)
      return { ...other, dropped: true, status: 'dropped', ready: false }
    if (other.dropped || !other.depends_on.includes(key)) return other
    return { ...other, depends_on: other.depends_on.filter((k) => k !== key) }
  })
  return {
    request: withItems(request, items),
    why: why(`drop ${key}`),
    stripped: [],
  }
}

/** A status for an item coming back into the plan, until the server says:
 * what its linked jobs make of it */
function statusOf(item: LedgerItem): LedgerItem['status'] {
  const has = (status: string) => item.attempts.some((a) => a.status === status)
  if (has('running')) return 'running'
  if (has('queued')) return 'queued'
  if (has('succeeded')) return 'done'
  return item.attempts.length ? 'failed' : 'planned'
}

/** Whether `from` reaches `to` along `depends_on`: an edge put back must
 * not close a cycle the agent's rewrite has made possible */
function reaches(items: LedgerItem[], from: string, to: string): boolean {
  const byKey = new Map(items.map((item) => [item.key, item]))
  const seen = new Set<string>()
  const stack = [from]
  while (stack.length) {
    const key = stack.pop() as string
    if (key === to) return true
    if (seen.has(key)) continue
    seen.add(key)
    stack.push(...(byKey.get(key)?.depends_on ?? []))
  }
  return false
}

function restore(
  request: LedgerRequest,
  { key, memory }: { key: string; memory?: DropMemory },
): Change | Refused {
  const stored = find(request, key) ?? memory?.item
  if (!stored) return no('item_gone')
  if (find(request, key) && !stored.dropped) return no('no_change')
  const others = request.items.filter((item) => item.key !== key)
  const live = new Set(others.filter((i) => !i.dropped).map((i) => i.key))
  // An item the server deleted comes back as a new one: a held deadline
  // would be restamped (re-arming a hold a note redirected) or refused as
  // too soon, so it returns unheld. The stored dropped copy rides as read.
  const fromMemory = !find(request, key)
  const item: LedgerItem = {
    ...stored,
    ...(fromMemory ? { review_until: null } : {}),
    dropped: false,
    status: statusOf(stored),
    depends_on: stored.depends_on.filter((dep) => live.has(dep)),
  }
  // Its old place if the item it sat before is still live, else the end
  const order = placeBefore(
    [...live],
    key,
    memory?.before && live.has(memory.before) ? memory.before : null,
  )
  const edges = new Set(
    (memory?.dependents ?? []).filter(
      (dep) => live.has(dep) && !reaches([...others, item], key, dep),
    ),
  )
  const byKey = new Map([...others, item].map((i) => [i.key, i]))
  const placed = order.map((k): LedgerItem => {
    const one = byKey.get(k) as LedgerItem
    return edges.has(k) && !one.depends_on.includes(key)
      ? { ...one, depends_on: [...one.depends_on, key] }
      : one
  })
  return {
    request: withItems(request, [
      ...placed,
      ...others.filter((i) => i.dropped),
    ]),
    why: why(`restore ${key}`),
    stripped: stored.depends_on.filter((dep) => !live.has(dep)),
  }
}

function move(
  request: LedgerRequest,
  { key, before }: { key: string; before: string | null },
): Change | Refused {
  const item = liveItem(request, key)
  if (isRefused(item)) return item
  if (before === key) return no('no_change')
  const live = request.items.filter((i) => !i.dropped)
  const keys = live.map((i) => i.key)
  if (before !== null && !keys.includes(before)) return no('anchor_gone')
  const order = placeBefore(keys, key, before)
  if (order.every((k, i) => k === keys[i])) return no('no_change')
  const byKey = new Map(live.map((i) => [i.key, i]))
  return {
    request: withItems(request, [
      ...order.map((k) => byKey.get(k) as LedgerItem),
      ...request.items.filter((i) => i.dropped),
    ]),
    why: why(`reorder ${key}`),
    stripped: [],
  }
}

const STATE_WHY: Record<LedgerRequest['state'], string> = {
  done: 'marked done',
  open: 'reopened',
  abandoned: 'abandoned',
}

function setState(
  request: LedgerRequest,
  { state, from }: Extract<Intent, { kind: 'state' }>,
): Change | Refused {
  if (request.state === state) return no('no_change')
  if (from !== undefined && request.state !== from) return no('moved_on')
  const next = { ...request, state }
  return {
    request: { ...next, items: settle(next, request.items) },
    why: STATE_WHY[state],
    stripped: [],
  }
}

/** `request` as `intent` would leave it, or why it cannot be applied */
export function project(
  intent: Intent,
  request: LedgerRequest,
): Change | Refused {
  switch (intent.kind) {
    case 'accept':
      return accept(request, intent)
    case 'drop':
      return drop(request, intent.key)
    case 'restore':
      return restore(request, intent)
    case 'move':
      return move(request, intent)
    case 'state':
      return setState(request, intent)
  }
}

/** An item as a write sends it: notes are the request-notes routes' to
 * write, and the server keeps the stored ones whatever a plan says */
function written(item: LedgerItem): WrittenItem {
  const copy: Partial<LedgerItem> = { ...item }
  delete copy.notes
  return copy as WrittenItem
}

/** The write that makes `intent` so over `request` as last read - the
 * whole plan back at the revision read, one thing changed - or the name of
 * the reason it cannot be made. Never `id`, `history`, a total or a note:
 * the server refuses a field it does not know. */
export function apply(
  intent: Intent,
  request: LedgerRequest,
): Applied | Refused {
  const change = project(intent, request)
  if (isRefused(change)) return change
  const next = change.request
  return {
    body: {
      title: next.title,
      goal: next.goal,
      state: next.state,
      revision: request.revision,
      why: change.why,
      items: next.items.map(written),
    },
    stripped: change.stripped,
  }
}

/** What the page shows while writes are still out: the intents in order
 * over the last read. One that no longer applies is passed over - its own
 * write will say so. A change of state is not shown before the server has
 * it. */
export function fold(request: LedgerRequest, intents: Intent[]): LedgerRequest {
  let view = request
  for (const intent of intents) {
    if (intent.kind === 'state') continue
    const change = project(intent, view)
    if (!isRefused(change)) view = change.request
  }
  return view
}

/** Items this page dropped that the server then deleted outright (no job
 * linked to them): shown with the dropped ones so they can be restored
 * until the page is left. */
export function withGhosts(
  request: LedgerRequest,
  memory: Record<string, DropMemory>,
): LedgerRequest {
  const held = new Set(request.items.map((item) => item.key))
  const ghosts = Object.values(memory)
    .filter(({ item }) => !held.has(item.key))
    .map(({ item }): LedgerItem => ({
      ...item,
      dropped: true,
      status: 'dropped',
      ready: false,
    }))
  return ghosts.length
    ? { ...request, items: [...request.items, ...ghosts] }
    : request
}
