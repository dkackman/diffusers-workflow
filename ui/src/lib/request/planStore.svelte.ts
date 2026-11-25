// The request the page holds, and the edits it has not yet seen written.
// `read` is the last thing the server said; `queue` the intents waiting to
// be written, one PUT each, one at a time. What the page draws is the
// queue folded over the read, so a refresh replaces the read without
// undoing an edit the user has just made. An intent is applied to the read
// only at the moment it is sent: a plan read earlier is never sent again.
import { api } from '../api'
import { ApiError } from '../apiErrors'
import type { RequestWriteBody } from '../apiRequest'
import type { UndoTarget } from '../destructive'
import type { LedgerItem, LedgerRequest } from '../types'
import {
  culprit,
  itemDiff,
  onlyNotes,
  stateCulprit,
  type Conflict,
} from './conflict'
import {
  apply,
  fold,
  isRefused,
  withGhosts,
  type DropMemory,
  type Intent,
  type Refusal,
} from './intents'
import { newerOf } from './refresh'

/** A write refused as stale is made again over a fresh read this many
 * times before the page gives up and says so */
export const MAX_RETRIES = 3

/** How an intent ended. `saved`: written. `unchanged`: the plan already
 * said so, nothing sent. `asked`: the item changed, so the page asks
 * first. `gone`: it no longer applies. */
export type Outcome = {
  status: 'saved' | 'unchanged' | 'asked' | 'gone'
  /** Dependencies a restore left behind: they are not in the plan */
  stripped: string[]
}

type Pending = {
  id: number
  intent: Intent
  /** The item as the user saw it when they asked */
  seen: LedgerItem | null
  /** The revision held when they asked */
  since: number
  tries: number
  /** A move still being made: not sent until committed */
  held: boolean
  settle: (outcome: Outcome) => void
  fail: (error: unknown) => void
}

export type PlanIo = {
  get: () => Promise<LedgerRequest>
  put: (body: RequestWriteBody) => Promise<LedgerRequest>
  /** Every time the read is replaced, with what it replaced */
  onread?: (previous: LedgerRequest | null, next: LedgerRequest) => void
  /** A write of the user's was saved: what is held after it, for a
   * resource to `adopt` so the edit is never marked as someone else's */
  onsaved?: (held: LedgerRequest) => void
  /** A read made to retry a stale write, after `onread`: the agent's
   * change it found, for a resource to `observe` as a quiet read so it is
   * marked, and not lost to the save's `adopt` */
  onquiet?: (read: LedgerRequest) => void
  /** An edit held back or refused after the page was left: no card of it
   * will be drawn */
  onunshown?: (conflict: Conflict) => void
}

/** The two calls a store makes for request `id` of workspace `ws` */
export function requestRoutes(
  ws: string,
  id: string,
): Pick<PlanIo, 'get' | 'put'> {
  return {
    get: () => api.getRequest(ws, id),
    put: (body) => api.putRequest(ws, id, body),
  }
}

const done = (status: Outcome['status'], stripped: string[] = []): Outcome => ({
  status,
  stripped,
})
const isStale = (error: unknown) =>
  error instanceof ApiError &&
  error.status === 409 &&
  error.code === 'stale_revision'
/** A refusal wrote nothing; any other failure may have */
const isRefusal = (error: unknown) =>
  error instanceof ApiError && error.status >= 400 && error.status < 500
const keyOf = (intent: Intent) => ('key' in intent ? intent.key : null)
/** A drop or an accept is a decision about the item as it looked: the
 * rest (a move, a restore) mean the same whatever it now holds */
const decides = (intent: Intent) =>
  intent.kind === 'drop' || intent.kind === 'accept'

/** The request a page opened last, as `workspace/id` */
let opened = ''

/** An Undo made on a request is not pressed once another one is open
 * (`undoToast`'s target) */
export const onThisRequest = (store: PlanStore): UndoTarget => ({
  now: () => (store.of === opened ? (store.read?.id ?? null) : null),
  away: 'Nothing undone: another request is open now.',
})

export class PlanStore {
  read = $state.raw<LedgerRequest | null>(null)
  queue = $state.raw<Pending[]>([])
  /** Edits not written, each waiting for the user to see or answer it */
  conflicts = $state.raw<Conflict[]>([])
  /** The write that a change of the user's went on top of */
  notice = $state.raw<{ entry: Conflict['entry'] } | null>(null)
  /** What this page's drops remember, by item key */
  memory = $state.raw<Record<string, DropMemory>>({})
  /** The request as the page shows it */
  view = $derived.by(() =>
    this.read
      ? withGhosts(
          fold(
            this.read,
            this.queue.map((entry) => entry.intent),
          ),
          this.memory,
        )
      : null,
  )

  /** The request this store is for, as `workspace/id` */
  readonly of: string
  /** The page has gone: what is queued is still written, and nothing is
   * said to the page */
  #left = false
  #io: PlanIo
  #sending = false
  #epoch = 0
  #ids = 0

  constructor(io: PlanIo, of = '') {
    this.#io = io
    this.of = of
    opened = of
  }

  /** The page is leaving this request. Every edit asked for, a staged move
   * included, is still written */
  leave() {
    this.#left = true
    this.commit()
  }

  /** Forget everything: the request is gone. Replies still on their way
   * are dropped. */
  reset() {
    this.#epoch++
    for (const entry of this.queue) entry.settle(done('unchanged'))
    this.queue = []
    this.conflicts = []
    this.notice = null
    this.memory = {}
    this.read = null
  }

  /** A read arrived: keep it unless what is held is newer. The queue is
   * untouched, so the view still shows every edit not yet written. */
  take(found: LedgerRequest) {
    const previous = this.read
    const next = newerOf(previous, found)
    if (next === previous) return
    this.read = next
    this.#forgetLive(next)
    if (!this.#left) this.#io.onread?.(previous, next)
  }

  /** Read the request again and keep it unless what is held is newer
   * (`take`). Resolves with what is then held; a reply for a request the
   * store has since been reset from, or that `isCurrent` says a later read
   * overtook, is not taken. `get` is called before any await, so a resource
   * loading through this tracks what it reads */
  async refresh(isCurrent?: () => boolean): Promise<LedgerRequest> {
    const epoch = this.#epoch
    const found = await this.#io.get()
    if (epoch !== this.#epoch || isCurrent?.() === false) return found
    this.take(found)
    return this.read ?? found
  }

  /** Queue an edit. Resolves with how it ended; rejects with the server's
   * error when the write failed for a reason that is not the plan moving. */
  push(intent: Intent): Promise<Outcome> {
    this.commit()
    return this.#enqueue(intent, false)
  }

  /** A move still in progress: it shows at once and replaces the one
   * staged before it for the same item, and nothing is sent until
   * `commit`. So a run of key presses is one write. */
  stage(intent: Extract<Intent, { kind: 'move' }>): Promise<Outcome> {
    const last = this.queue.at(-1)
    const again =
      last?.held &&
      last.intent.kind === 'move' &&
      last.intent.key === intent.key
    if (last && again) {
      this.queue = this.queue.slice(0, -1)
      last.settle(done('unchanged'))
    } else this.commit()
    return this.#enqueue(intent, true)
  }

  /** Send the staged move */
  commit() {
    const last = this.queue.at(-1)
    if (!last?.held) return
    last.held = false
    void this.#pump()
  }

  /** Drop an item, remembering where it was and what depended on it */
  async drop(key: string): Promise<Outcome> {
    const view = this.view
    const item = view?.items.find((i) => i.key === key)
    if (!view || !item || item.dropped) return done('unchanged')
    const live = view.items.filter((i) => !i.dropped)
    const at = live.findIndex((i) => i.key === key)
    const remembered: DropMemory = {
      item,
      before: live[at + 1]?.key ?? null,
      dependents: live
        .filter((i) => i.depends_on.includes(key))
        .map((i) => i.key),
    }
    this.memory = { ...this.memory, [key]: remembered }
    try {
      const outcome = await this.push({ kind: 'drop', key })
      if (outcome.status !== 'saved') this.#forget(key)
      return outcome
    } catch (error) {
      // What is not a refusal may have been applied with its reply lost:
      // remembered until a read shows the item live
      if (isRefusal(error)) this.#forget(key)
      throw error
    }
  }

  /** Put a dropped item back: where it was, with what depended on it, when
   * this page dropped it; else last, with the dependencies still live */
  async restore(key: string): Promise<Outcome> {
    const outcome = await this.push({
      kind: 'restore',
      key,
      memory: this.memory[key],
    })
    if (outcome.status !== 'asked') this.#forget(key)
    return outcome
  }

  /** The user's answer to a held-back edit: make it over the item as it
   * now is. Asked again only if the item changes once more. */
  again(id: number): Promise<Outcome> {
    const conflict = this.conflicts.find((c) => c.id === id)
    this.dismiss(id)
    if (!conflict) return Promise.resolve(done('unchanged'))
    const { intent } = conflict
    return intent.kind === 'drop' ? this.drop(intent.key) : this.push(intent)
  }

  dismiss(id: number) {
    this.conflicts = this.conflicts.filter((c) => c.id !== id)
  }

  /** A read that shows a remembered item live, with no drop of it still to
   * be written, ends the memory: the drop did not happen, or it has been
   * put back since, and its old dependents are no longer this page's */
  #forgetLive(read: LedgerRequest) {
    for (const key of Object.keys(this.memory)) {
      const live = read.items.some((i) => i.key === key && !i.dropped)
      const dropping = this.queue.some(
        (e) => e.intent.kind === 'drop' && e.intent.key === key,
      )
      if (live && !dropping) this.#forget(key)
    }
  }

  #forget(key: string) {
    if (!(key in this.memory)) return
    this.memory = Object.fromEntries(
      Object.entries(this.memory).filter(([held]) => held !== key),
    )
  }

  #enqueue(intent: Intent, held: boolean): Promise<Outcome> {
    const key = keyOf(intent)
    const seen = this.view?.items.find((item) => item.key === key) ?? null
    return new Promise<Outcome>((settle, fail) => {
      const entry: Pending = {
        id: ++this.#ids,
        intent,
        seen,
        since: this.read?.revision ?? 0,
        tries: 0,
        held,
        settle,
        fail,
      }
      this.queue = [...this.queue, entry]
      void this.#pump()
    })
  }

  #head(): Pending | null {
    const first = this.queue[0]
    return first && !first.held ? first : null
  }

  async #pump() {
    if (this.#sending) return
    this.#sending = true
    const epoch = this.#epoch
    try {
      for (let next = this.#head(); next; next = this.#head()) {
        await this.#send(next, epoch)
        if (epoch !== this.#epoch) break
      }
    } finally {
      this.#sending = false
    }
    // A reset mid-write left this loop: what was queued since is not sent
    if (epoch !== this.#epoch && this.#head()) void this.#pump()
  }

  #finish(entry: Pending, outcome: Outcome) {
    this.queue = this.queue.filter((e) => e !== entry)
    entry.settle(outcome)
  }

  #reject(entry: Pending, error: unknown) {
    this.queue = this.queue.filter((e) => e !== entry)
    entry.fail(error)
  }

  #conflict(
    entry: Pending,
    basis: LedgerRequest,
    kind: Conflict['kind'],
    refusal: Refusal | null,
    changes: string[],
  ) {
    const key = keyOf(entry.intent)
    const conflict: Conflict = {
      id: entry.id,
      kind,
      intent: entry.intent,
      refusal,
      key,
      title: entry.seen?.title ?? key ?? 'this request',
      // A state moved on may have been read before this write was asked
      // for: the write that set the state names it, however long ago
      entry:
        refusal === 'moved_on'
          ? stateCulprit(basis.history)
          : culprit(basis.history, entry.since),
      changes,
    }
    this.conflicts = [...this.conflicts, conflict]
    if (this.#left) this.#io.onunshown?.(conflict)
    this.#finish(entry, done(kind === 'changed' ? 'asked' : 'gone'))
  }

  /** What changed about the item since the user decided, when the intent
   * is a decision about it */
  #changes(entry: Pending, basis: LedgerRequest): string[] {
    if (!decides(entry.intent) || !entry.seen) return []
    const now = basis.items.find((item) => item.key === entry.seen?.key)
    return now ? itemDiff(entry.seen, now) : []
  }

  /** The write for `entry` over `basis`, or null when it ended here: the
   * plan already says so, it no longer applies, or its item changed */
  #prepare(entry: Pending, basis: LedgerRequest) {
    const applied = apply(entry.intent, basis)
    if (isRefused(applied)) {
      if (applied.refusal === 'no_change')
        this.#finish(entry, done('unchanged'))
      else this.#conflict(entry, basis, 'gone', applied.refusal, [])
      return null
    }
    const changes = this.#changes(entry, basis)
    if (!changes.length) return applied
    this.#conflict(entry, basis, 'changed', null, changes)
    return null
  }

  async #send(entry: Pending, epoch: number) {
    const basis = this.read
    if (!basis) return this.#finish(entry, done('unchanged'))
    const applied = this.#prepare(entry, basis)
    if (!applied) return
    let saved: LedgerRequest
    try {
      saved = await this.#io.put(applied.body)
    } catch (error) {
      if (epoch === this.#epoch) await this.#refused(entry, error, epoch)
      return
    }
    if (epoch !== this.#epoch) return
    this.take(saved)
    if (!this.#left) this.#io.onsaved?.(this.read ?? saved)
    this.#noticeOver(entry, saved)
    this.#finish(entry, done('saved', applied.stripped))
  }

  /** Say that a retried write went on top of someone else's change */
  #noticeOver(entry: Pending, saved: LedgerRequest) {
    const { tries, since } = entry
    if (!tries || onlyNotes(saved.history, since, saved.revision)) return
    this.notice = { entry: culprit(saved.history, since, saved.revision) }
  }

  /** A write the server did not take. Stale: read again and leave the
   * entry at the head, to be applied to what came back - unless it has
   * been tried enough. Anything else ends it with the error. */
  async #refused(entry: Pending, error: unknown, epoch: number) {
    if (!isStale(error) || ++entry.tries > MAX_RETRIES)
      return this.#reject(entry, error)
    try {
      const fresh = await this.#io.get()
      if (epoch === this.#epoch) {
        this.take(fresh)
        if (!this.#left) this.#io.onquiet?.(fresh)
      }
    } catch (failure) {
      if (epoch === this.#epoch) this.#reject(entry, failure)
    }
  }
}
