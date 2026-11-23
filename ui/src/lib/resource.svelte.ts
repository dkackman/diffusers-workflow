// The one way a page loads server data. A resource says how it stays fresh
// - the change kinds it watches (`watch`, `[]` for data that never
// changes), or a `live` poll for machine state, with `alsoLive` adding a
// timer beside the watch while there is something in flight - and every
// trigger goes through one `run`:
//   - its dependencies changed (what `load` reads before its first await,
//     and the workspace): a loud reload, `status` 'loading', `data` kept
//     unless the workspace changed;
//   - a watched kind moved, `connection.epoch` rose, a timer ticked: a
//     quiet reload, never 'loading', and a failure keeps `data` and
//     `status` and sets `stale`;
//   - `reload()`: loud.
// A reply overtaken by a later run is dropped (latestOnly). Timers go
// through `poll`, the feed through `changes(workspace).watch`. LoadState
// (ui/LoadState.svelte) renders one.
import { untrack } from 'svelte'
import { changes, type ChangeKind } from './changes.svelte'
import { connection } from './connection.svelte'
import { latestOnly } from './latest'
import { poll } from './poll'
import { workspace as workspaceState } from './workspaceState.svelte'

export type Freshness =
  | {
      watch: readonly ChangeKind[]
      /** A timer beside the watch, at the interval it returns while it is
       * not null: a download or a job in flight */
      alsoLive?: () => number | null
      live?: never
    }
  | { live: { everyMs: number }; watch?: never; alsoLive?: never }

export type ResourceOptions<T> = Freshness & {
  /** Which workspace the data belongs to; `workspace.current` by default */
  workspace?: () => string
  /** What is held when a reply lands; the reply by default */
  merge?: (held: T | undefined, incoming: T) => T
}

export interface Resource<T> {
  readonly status: 'loading' | 'ready' | 'error'
  readonly data: T | undefined
  readonly error: unknown
  readonly stale: boolean
  /** A loud reload: Retry */
  reload(): Promise<void>
}

type Status = Resource<unknown>['status']
type Run = (quiet: boolean) => Promise<void>

/** The triggers that are not dependencies: the feed, the epoch, timers.
 * Each calls `run(true)` */
function keepFresh(
  options: Freshness,
  workspaceOf: () => string,
  run: Run,
): void {
  const quietly = () => untrack(() => run(true))
  const kinds = options.watch ?? []
  if (kinds.length)
    $effect(() => changes(workspaceOf()).watch(kinds, () => void quietly()))
  if (options.live) {
    const ms = options.live.everyMs
    $effect(() => poll(quietly, ms, { immediate: false }))
  }
  const alsoLive = options.alsoLive
  if (alsoLive)
    $effect(() => {
      const ms = alsoLive()
      return ms === null ? undefined : poll(quietly, ms, { immediate: false })
    })
  let seen = untrack(() => connection.epoch)
  $effect(() => {
    const epoch = connection.epoch
    if (epoch === seen) return
    seen = epoch
    void quietly()
  })
}

/** Load `load` now and whenever it should be read again (above). Call it
 * during component init: it registers its effects there */
export function resource<T>(
  load: () => Promise<T>,
  options: ResourceOptions<T>,
): Resource<T> {
  let status = $state<Status>('loading')
  let data = $state.raw<T | undefined>(undefined)
  let error = $state.raw<unknown>(undefined)
  let stale = $state(false)
  const begin = latestOnly()
  const workspaceOf = options.workspace ?? (() => workspaceState.current)
  const merge =
    options.merge ?? ((_held: T | undefined, incoming: T) => incoming)

  function landed(value: T) {
    data = merge(
      untrack(() => data),
      value,
    )
    status = 'ready'
    error = undefined
    stale = false
  }
  function failed(e: unknown, quiet: boolean) {
    error = e
    // A quiet run that overtook a loud one answers for it: never left loading
    if (quiet && status !== 'loading') stale = true
    else status = 'error'
  }

  // `load` is called before any await, so inside the dependency effect what
  // it reads first is tracked
  async function run(quiet: boolean): Promise<void> {
    const current = begin()
    if (!quiet) status = 'loading'
    try {
      const value = await load()
      if (current()) landed(value)
    } catch (e) {
      if (current()) failed(e, quiet)
    }
  }

  let loadedIn: string | undefined
  $effect(() => {
    const ws = workspaceOf()
    if (loadedIn !== undefined && ws !== loadedIn) data = undefined
    loadedIn = ws
    void run(false)
  })
  keepFresh(options, workspaceOf, run)

  return {
    get status() {
      return status
    },
    get data() {
      return data
    },
    get error() {
      return error
    },
    get stale() {
      return stale
    },
    reload: () => untrack(() => run(false)),
  }
}
