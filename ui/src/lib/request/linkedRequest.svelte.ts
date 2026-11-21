// The request a job or a gallery file is linked to, read for a page that is
// not the request's own: the job page's title and its note box share one
// read, the gallery's box has its own. A read after an earlier one tells
// of every note the agent answered in between (answeredNow), once per note
// for this object's life; the first read tells of none.
import { untrack } from 'svelte'
import { ApiError, api } from '../api'
import { go } from '../router.svelte'
import { notify } from '../toast'
import type { LedgerRequest } from '../types'
import { answeredNow, answeredText } from './answered'
import { noteAnchor } from './noteState'
import { newerOf } from './refresh'

export class LinkedRequest {
  /** The request as last read; null until the first read answers */
  request = $state<LedgerRequest | null>(null)
  /** The server said 404: the request, or its workspace, was deleted */
  gone = $state(false)
  readonly workspace: string
  readonly id: string
  #asked = 0
  // Not reactive: which notes were told of is no part of what renders
  #told: Record<string, true> = {}

  constructor(workspace: string, id: string) {
    this.workspace = workspace
    this.id = id
  }

  /** Read the request again. A reply overtaken by a later read is dropped;
   * any failure but a 404 keeps what is held */
  async read(): Promise<void> {
    const asked = ++this.#asked
    let found: LedgerRequest
    try {
      found = await api.getRequest(this.workspace, this.id)
    } catch (e) {
      if (asked === this.#asked && e instanceof ApiError && e.status === 404)
        this.gone = true
      return
    }
    if (asked !== this.#asked) return
    const previous = untrack(() => this.request)
    const next = newerOf(previous, found)
    this.request = next
    this.gone = false
    if (previous)
      for (const note of answeredNow(previous, next)) this.#tell(next, note.id)
  }

  #tell(request: LedgerRequest, id: string) {
    if (this.#told[id]) return
    this.#told[id] = true
    notify.news(answeredText(request), {
      label: 'Show the note',
      onClick: () => this.#show(id),
    })
  }

  // The note where this page shows it; else the request page, which has it
  #show(id: string) {
    const note = document.getElementById(noteAnchor(id))
    if (!note) return go('ws', this.workspace, 'requests', this.id)
    note.scrollIntoView?.({ block: 'center' })
    note.focus({ preventScroll: true })
  }
}
