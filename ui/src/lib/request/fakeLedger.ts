// A ledger for the request tests that keeps the server's contract where it
// matters to an edit: a write at any revision but the current one is a 409
// `stale_revision`; an item sent as dropped (or left out) stays as dropped
// when a job links to it and is deleted when none does; dropped items sit
// after the live ones; every write adds a history entry.
import { ApiError } from '../apiErrors'
import type { RequestWriteBody } from '../apiRequest'
import type { LedgerItem, LedgerRequest } from '../types'
import { entry } from './fixtures'

function merge(current: LedgerRequest, body: RequestWriteBody): LedgerItem[] {
  const live = body.items
    .filter((item) => !item.dropped)
    .map((item) => ({
      ...item,
      comment: item.comment
        ? { ...item.comment, at: item.comment.at ?? 'now' }
        : null,
    }))
  const listed = new Set(live.map((item) => item.key))
  const dropped = current.items
    .filter((item) => !listed.has(item.key) && item.attempts.length > 0)
    .map((item): LedgerItem => ({ ...item, dropped: true, status: 'dropped' }))
  return [...live, ...dropped]
}

export class FakeLedger {
  current: LedgerRequest
  puts: RequestWriteBody[] = []
  gets = 0
  #gate: Promise<void> | null = null

  constructor(initial: LedgerRequest) {
    this.current = initial
  }

  #write(next: Partial<LedgerRequest>, by: 'agent' | 'user', why: string) {
    const revision = this.current.revision + 1
    this.current = {
      ...this.current,
      ...next,
      revision,
      history: [...this.current.history, entry(revision, by, why)],
    }
    return this.current
  }

  /** The agent rewrites the plan between the page's read and its write */
  agent(why: string, change: (items: LedgerItem[]) => LedgerItem[]) {
    return this.#write({ items: change(this.current.items) }, 'agent', why)
  }

  /** Hold every write from here until the returned function is called */
  hold() {
    let open: () => void = () => {}
    this.#gate = new Promise((resolve) => (open = resolve))
    return () => {
      this.#gate = null
      open()
    }
  }

  async get() {
    this.gets++
    return this.current
  }

  async put(body: RequestWriteBody): Promise<LedgerRequest> {
    this.puts.push(body)
    if (this.#gate) await this.#gate
    const { current } = this
    if (body.revision !== current.revision)
      throw new ApiError(
        `the request is at revision ${current.revision}`,
        409,
        'stale_revision',
      )
    const comment =
      body.comment === undefined
        ? current.comment
        : body.comment && { ...body.comment, at: 'now' }
    return this.#write(
      {
        title: body.title,
        goal: body.goal,
        state: body.state ?? current.state,
        comment,
        items: merge(current, body),
      },
      'user',
      body.why,
    )
  }
}

export const fakeLedger = (initial: LedgerRequest) => new FakeLedger(initial)
