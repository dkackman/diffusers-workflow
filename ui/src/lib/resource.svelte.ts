// The one way a page loads server data. A resource says how it stays fresh
// - the change kinds it watches (`watch`, `[]` for data that never
// changes), or a `live` poll for machine state, with `alsoLive` adding a
// timer beside the watch while there is something in flight - and every
// trigger goes through one `run`:
//   - its dependencies changed (what `load` reads before its first await,
//     and the workspace): a new question, a loud reload, `status`
//     'loading', `data` kept unless the workspace changed - and dropped if
//     the new question's load fails, so the old answer never stands in for
//     it (unless the failure is an outage or a lock, the banner's to say);
//   - a watched kind moved, `connection.epoch` rose, a timer ticked: a
//     quiet reload, never 'loading', and a failure keeps `data` and
//     `status` and sets `stale`;
//   - `reload()`: loud.
// A `lazy` resource skips the first load: it reads only once a watched
// kind moves or the epoch rises. With `rows`, a quiet reload sets
// `changed` to the keys whose value is new or differs from the held read,
// for a page to mark them (`.row-changed`) until CHANGED_MARK_MS has
// passed. A reply overtaken by a later run is dropped (latestOnly). Timers
// go through `poll`, the feed through `changes(workspace).watch`. LoadState
// (ui/LoadState.svelte) renders one.
import { untrack } from 'svelte'
import { describeError } from './apiErrors'
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
  /** The data as keyed rows, for `changed`: a request's id, a file's name */
  rows?: (data: T) => Iterable<readonly [key: string, value: unknown]>
  /** Load only on a watched move or an epoch rise, never at first */
  lazy?: boolean
}

export interface Resource<T> {
  readonly status: 'loading' | 'ready' | 'error'
  readonly data: T | undefined
  readonly error: unknown
  readonly stale: boolean
  /** The rows the last quiet reload changed or added, by key; empty after
   * a loud load, and again CHANGED_MARK_MS after it was set */
  readonly changed: ReadonlySet<string>
  /** A loud reload: Retry */
  reload(): Promise<void>
}

/** How long a changed row stays marked; `.row-changed` fades over it */
export const CHANGED_MARK_MS = 4000

type Status = Resource<unknown>['status']
type Run = (quiet: boolean) => Promise<void>
type Rows<T> = NonNullable<ResourceOptions<T>['rows']>

/** The keys of `next`'s rows that `held` lacks or holds another value for */
export function changedKeys<T>(
  rows: Rows<T>,
  held: T,
  next: T,
): ReadonlySet<string> {
  const before: Record<string, string> = Object.fromEntries(
    Array.from(rows(held), ([key, value]) => [key, JSON.stringify(value)]),
  )
  const keys = Array.from(rows(next))
    .filter(
      ([key, value]) =>
        !Object.hasOwn(before, key) || before[key] !== JSON.stringify(value),
    )
    .map(([key]) => key)
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- a value, never mutated
  return new Set(keys)
}

const NONE: ReadonlySet<string> = new Set()

/** `changed`, and the timer that clears it */
function changeMarks() {
  let changed = $state.raw(NONE)
  let timer: ReturnType<typeof setTimeout> | undefined
  $effect(() => () => clearTimeout(timer))
  return {
    get changed() {
      return changed
    },
    /** Mark `keys`; a reload that changed nothing leaves the marks fading */
    mark(keys: ReadonlySet<string>) {
      if (!keys.size) return
      changed = keys
      clearTimeout(timer)
      timer = setTimeout(() => (changed = NONE), CHANGED_MARK_MS)
    },
    clear() {
      clearTimeout(timer)
      if (changed.size) changed = NONE
    },
  }
}

/** A failure that is not an answer to the question: the connection
 * banner says it, and what is held stays under it */
function bannerSays(e: unknown): boolean {
  const kind = describeError(e).kind
  return kind === 'offline' || kind === 'locked'
}

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

/** What a resource holds, and how a reply or a failure changes it */
class Held<T> {
  status = $state<Status>('loading')
  data = $state.raw<T | undefined>(undefined)
  error = $state.raw<unknown>(undefined)
  stale = $state(false)
  readonly marks = changeMarks()
  /** The question `data` answers: a count of dependency changes */
  heldFor = -1

  constructor(
    private readonly merge: (held: T | undefined, incoming: T) => T,
    private readonly rows: Rows<T> | undefined,
  ) {}

  landed(value: T, quiet: boolean, question: number) {
    const held = untrack(() => this.data)
    const next = this.merge(held, value)
    if (!quiet) this.marks.clear()
    else if (this.rows && held !== undefined && this.heldFor === question)
      this.marks.mark(changedKeys(this.rows, held, next))
    this.data = next
    this.heldFor = question
    this.status = 'ready'
    this.error = undefined
    this.stale = false
  }

  failed(e: unknown, quiet: boolean, question: number) {
    this.error = e
    // Another question's answer is not this one's stale data
    const dropped = this.heldFor !== question && !bannerSays(e)
    if (dropped) {
      this.forget()
      this.stale = false
    }
    // A quiet run that overtook a loud one answers for it: never left loading
    if (quiet && !dropped && this.status !== 'loading') this.stale = true
    else this.status = 'error'
  }

  forget() {
    this.data = undefined
    this.marks.clear()
  }
}

/** Load `load` now and whenever it should be read again (above). Call it
 * during component init: it registers its effects there */
export function resource<T>(
  load: () => Promise<T>,
  options: ResourceOptions<T>,
): Resource<T> {
  const held = new Held<T>(
    options.merge ?? ((_held: T | undefined, incoming: T) => incoming),
    options.rows,
  )
  const begin = latestOnly()
  const workspaceOf = options.workspace ?? (() => workspaceState.current)
  // Which question is asked: a count of dependency changes
  let asked = 0

  // `load` is called before any await, so inside the dependency effect what
  // it reads first is tracked
  async function run(quiet: boolean): Promise<void> {
    const current = begin()
    const question = asked
    if (!quiet) held.status = 'loading'
    try {
      const value = await load()
      if (current()) held.landed(value, quiet, question)
    } catch (e) {
      if (current()) held.failed(e, quiet, question)
    }
  }

  let loadedIn: string | undefined
  $effect(() => {
    const ws = workspaceOf()
    if (loadedIn !== undefined && ws !== loadedIn) held.forget()
    loadedIn = ws
    asked++
    if (!options.lazy) void run(false)
  })
  keepFresh(options, workspaceOf, run)

  return {
    get status() {
      return held.status
    },
    get data() {
      return held.data
    },
    get error() {
      return held.error
    },
    get stale() {
      return held.stale
    },
    get changed() {
      return held.marks.changed
    },
    reload: () => untrack(() => run(false)),
  }
}
