// What moved on the server: one shared read of a workspace's change
// counters (GET /api/workspaces/{name}/changes) per workspace, asked every
// CHANGES_POLL_MS while something watching it is on screen and the tab is
// visible (poll.ts pauses it while hidden and asks once on return). A view
// of server data that is not live state reloads from `watch` - never on a
// timer of its own (the table below; ui/CLAUDE.md). Modelled on the
// notes channel (request/listening.svelte.ts): counted holders, a reply
// overtaken by a later ask dropped, and a failed read keeps the counters
// it had, so the next good read catches up.
//
// Which view keeps itself fresh how:
//   view                        live poll                       feed watch
//   App shell (App.svelte)      status, 5 s, also while hidden  -
//                               (statusPoll.ts)
//   Sidebar workspace list      -                               workspaces: clears the
//                                                               cached list, reads again
//   Requests (RequestsPage)     -                               requests: quiet reload
//   One request (RequestPage)   5 s while a job of it is        requests, only while
//                               queued or running, 20 s while   closed; reads keep the
//                               open (request/refresh.ts)       higher revision (newerOf)
//   Linked notes                20 s                            -
//   Jobs (JobsPage)             3 s                             -
//   Models (ModelsPage)         2 s while a download or a       -
//                               diffusers update runs
//   Server status (ServerPage)  health and memory, 5 s          -
import { api } from './api'
import { poll } from './poll'
import type { ChangeCounts, WorkspaceChanges } from './types'

export const CHANGES_POLL_MS = 5000

/** A kind of data a page lists, or `workspaces`: the server-wide counter
 * of workspace create, delete and note writes */
export type ChangeKind = keyof ChangeCounts | 'workspaces'

interface Watcher {
  kinds: readonly ChangeKind[]
  fn: () => void
}

const count = (snapshot: WorkspaceChanges, kind: ChangeKind) =>
  kind === 'workspaces' ? snapshot.workspaces : snapshot.kinds[kind]

export class ChangeFeed {
  #workspace: string
  // Not reactive: who listens is bookkeeping, never shown
  // eslint-disable-next-line svelte/prefer-svelte-reactivity
  #watchers = new Set<Watcher>()
  #stop: (() => void) | null = null
  #asked = 0
  // The counters as last read: what the next read is compared to. Kept
  // while nothing watches, so a page that comes back reloads once for what
  // moved meanwhile rather than missing it
  #seen: WorkspaceChanges | null = null

  constructor(workspace: string) {
    this.#workspace = workspace
  }

  get watching(): number {
    return this.#watchers.size
  }

  /** Call `fn` when any of `kinds` moves, or the epoch does (a server
   * restart, the workspace made again: everything changed). Holds the feed
   * until the returned function is called. Written to be returned from an
   * $effect */
  watch(kinds: readonly ChangeKind[], fn: () => void): () => void {
    const watcher: Watcher = { kinds, fn }
    this.#watchers.add(watcher)
    if (this.#stop === null)
      this.#stop = poll(() => this.refresh(), CHANGES_POLL_MS)
    let held = true
    return () => {
      if (!held) return
      held = false
      this.#watchers.delete(watcher)
      if (this.#watchers.size > 0 || this.#stop === null) return
      this.#stop()
      this.#stop = null
    }
  }

  /** Read the counters now and call the watchers of what moved. A reply
   * overtaken by a later read is dropped; a failed one changes nothing */
  async refresh(): Promise<void> {
    const asked = ++this.#asked
    let next: WorkspaceChanges
    try {
      next = await api.workspaceChanges(this.#workspace)
    } catch {
      return
    }
    if (asked !== this.#asked) return
    const seen = this.#seen
    this.#seen = next
    // The first read is the baseline: each page loaded itself already
    if (!seen) return
    const restarted = seen.epoch !== next.epoch
    for (const watcher of [...this.#watchers]) {
      const moved =
        restarted ||
        watcher.kinds.some((kind) => count(next, kind) !== count(seen, kind))
      if (!moved) continue
      try {
        watcher.fn()
      } catch {
        // one watcher failing does not keep the news from the rest
      }
    }
  }
}

// Not reactive: a feed is made on first ask, and what it calls is the news
const feeds: Record<string, ChangeFeed> = {}

/** The shared change feed for `workspace` */
export function changes(workspace: string): ChangeFeed {
  return (feeds[workspace] ??= new ChangeFeed(workspace))
}
