// A ledger for the request tests that keeps the server's contract where it
// matters to an edit: a write at any revision but the current one is a 409
// `stale_revision`; an item sent as dropped (or left out) stays as dropped
// when a job links to it and is deleted when none does; dropped items sit
// after the live ones; every write adds a history entry. Notes are written
// only through the note calls: a plan write keeps the stored ones.
import { ApiError } from '../apiErrors'
import type { RequestWriteBody } from '../apiRequest'
import type { LedgerItem, LedgerNote, LedgerRequest, NoteAdd } from '../types'
import { entry } from './fixtures'

function merge(current: LedgerRequest, body: RequestWriteBody): LedgerItem[] {
  const stored = new Map(current.items.map((item) => [item.key, item.notes]))
  const live = body.items
    .filter((item) => !item.dropped)
    .map((item): LedgerItem => ({ ...item, notes: stored.get(item.key) ?? [] }))
  const listed = new Set(live.map((item) => item.key))
  const dropped = current.items
    .filter((item) => !listed.has(item.key) && item.attempts.length > 0)
    .map((item): LedgerItem => ({ ...item, dropped: true, status: 'dropped' }))
  return [...live, ...dropped]
}

export class FakeLedger {
  current: LedgerRequest
  puts: RequestWriteBody[] = []
  added: NoteAdd[] = []
  gets = 0
  #gate: Promise<void> | null = null

  constructor(initial: LedgerRequest) {
    this.current = initial
  }

  #write(next: Partial<LedgerRequest>, by: 'agent' | 'user', why: string) {
    const revision = this.current.revision + 1
    const written = entry(revision, by, why)
    this.current = {
      ...this.current,
      ...next,
      revision,
      history: [...this.current.history, written],
      ...(by === 'agent' ? { agent_wrote_at: written.at } : {}),
    }
    return this.current
  }

  /** The agent rewrites the plan between the page's read and its write */
  agent(why: string, change: (items: LedgerItem[]) => LedgerItem[]) {
    return this.#write({ items: change(this.current.items) }, 'agent', why)
  }

  /** A write made by the user outside the page's store (a note route) */
  elsewhere(why: string) {
    return this.#write({}, 'user', why)
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
    return this.#write(
      {
        title: body.title,
        goal: body.goal,
        state: body.state ?? current.state,
        items: merge(current, body),
      },
      'user',
      body.why,
    )
  }

  /** POST request-notes: refused, as the server does, on an item that is
   * dropped or not in the plan */
  async addNote(body: NoteAdd) {
    this.added.push(body)
    const { current } = this
    const key = body.item ?? null
    const target =
      key === null ? null : current.items.find((i) => i.key === key)
    if (key !== null && !target)
      throw new ApiError(`no item ${key}`, 422, 'unknown_item')
    if (target?.dropped)
      throw new ApiError(`item ${key} was dropped`, 422, 'item_dropped')
    const at = entry(current.revision + 1, 'user', '').at
    const note: LedgerNote = {
      id: `n${current.next_note}`,
      text: body.text,
      created: at,
      at,
      ...(body.refs?.length
        ? {
            refs: body.refs.map((r) => ({
              item: r.item ?? key ?? '',
              ...(r.job ? { job: r.job } : {}),
              ...(r.output ? { output: r.output } : {}),
            })),
          }
        : {}),
    }
    const next =
      key === null
        ? { notes: [...current.notes, note] }
        : {
            items: current.items.map((i) =>
              i.key === key ? { ...i, notes: [...i.notes, note] } : i,
            ),
          }
    const written = this.#write(
      { ...next, next_note: current.next_note + 1 },
      'user',
      'note added',
    )
    return { note, revision: written.revision }
  }
}

export const fakeLedger = (initial: LedgerRequest) => new FakeLedger(initial)
