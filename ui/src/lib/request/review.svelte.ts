// The countdowns of items held for review. One shared one-second ticker,
// running only while a hold is on screen and the tab is visible (the
// hold() pattern of NotesChannel, listening.svelte.ts; a `poll`). A countdown is
// anchored to the moment the request was read plus the server's
// `seconds_left`: the browser's clock is never compared with
// `review.until`, so a skewed clock cannot make an item look due early.
// When a countdown reaches zero the page reads the request once more, so
// the server's own `due` replaces the local guess.
import { poll } from '../poll'
import type { LedgerRequest } from '../types'

export const TICK_MS = 1000

const visible = () =>
  typeof document === 'undefined' || document.visibilityState !== 'hidden'

type Listener = (now: number) => void

export class Ticker {
  /** Date.now() at the last tick */
  now = $state(Date.now())
  #holds: { listener?: Listener }[] = []
  #stop: (() => void) | null = null

  /** Held and on screen: a hidden tab does not tick */
  get running() {
    return this.#stop !== null && visible()
  }

  /** Something on screen counts down: tick, calling `listener` each time,
   * until the returned function says it has gone. Written to be returned
   * from an $effect */
  hold(listener?: Listener): () => void {
    const hold = { listener }
    this.#holds.push(hold)
    if (this.#holds.length === 1) this.#stop = poll(this.#tick, TICK_MS)
    else if (this.running) listener?.(this.now)
    return () => {
      const at = this.#holds.indexOf(hold)
      if (at < 0) return
      this.#holds.splice(at, 1)
      if (this.#holds.length) return
      this.#stop?.()
      this.#stop = null
    }
  }

  #tick = () => {
    this.now = Date.now()
    for (const hold of [...this.#holds]) hold.listener?.(this.now)
  }
}

/** The one ticker every countdown on the page shares */
export const ticker = new Ticker()

type Anchor = { deadline: number; until: string }

/** One request's countdowns: `read` with every read the page keeps, `left`
 * for a row's seconds, `hold` while one is on screen */
export class ReviewCountdown {
  #anchors = $state<Record<string, Anchor>>({})
  #readAt = $state(0)
  /** The deadlines already re-read for, as `key@until` */
  #asked: Record<string, true> = {}
  #refresh: () => void
  #clock: Ticker

  /** `refresh` re-reads the request through the page's own path */
  constructor(refresh: () => void, clock: Ticker = ticker) {
    this.#refresh = refresh
    this.#clock = clock
  }

  /** A read of the request the page now holds, made at `at` (ms). Only a
   * holding row counts down; a redirected or due one has no digits */
  read(request: LedgerRequest, at = Date.now()) {
    const anchors: Record<string, Anchor> = {}
    for (const { key, review } of request.items)
      if (review?.state === 'holding')
        anchors[key] = {
          deadline: at + review.seconds_left * 1000,
          until: review.until,
        }
    this.#anchors = anchors
    this.#readAt = at
  }

  /** Whole seconds left on `key`'s countdown, null when it has none */
  left(key: string): number | null {
    const anchor = this.#anchors[key]
    if (!anchor) return null
    // a tick older than the read counts as the read
    const now = Math.max(this.#clock.now, this.#readAt)
    return Math.max(0, Math.ceil((anchor.deadline - now) / 1000))
  }

  /** A countdown is on screen: tick until the returned function is called */
  hold(): () => void {
    return this.#clock.hold((now) => this.#check(now))
  }

  /** A countdown reached zero: read the request again, once per deadline */
  #check(now: number) {
    let due = false
    for (const [key, anchor] of Object.entries(this.#anchors)) {
      const asked = `${key}@${anchor.until}`
      if (now < anchor.deadline || this.#asked[asked]) continue
      this.#asked[asked] = true
      due = true
    }
    if (due) this.#refresh()
  }
}
