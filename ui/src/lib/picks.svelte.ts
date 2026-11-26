import { SvelteMap, SvelteSet } from 'svelte/reactivity'
import { action } from './action.svelte'
import { describeError } from './apiErrors'
import type { UndoTarget } from './destructive'
import { leafName } from './names'
import { notify } from './toast'

/** A star Undo over files whose star someone else changed meanwhile */
export const STAR_CHANGED = 'Nothing undone: the star was changed since.'

/** One file's star: the button and the S key, in the Gallery and the job
 * viewer. A star is undone, not asked about, and the Undo leaves a star
 * `read` no longer shows as this set it (another writer's). One action per
 * file: a second press on a file in flight is dropped, while the next
 * file's goes through (S down a run, faster than writes) */
export function starAction(
  set: (name: string, next: boolean) => Promise<unknown>,
  opts: {
    /** The page's word on a file's star; null while it is not known, when
     * a toggle would be a guess and does nothing */
    read: (name: string) => boolean | null
    /** Told after each write lands, so a page can hold it as its own */
    applied: (name: string, next: boolean) => void
    target?: UndoTarget
  },
) {
  const put = async (name: string, next: boolean) => {
    await set(name, next)
    opts.applied(name, next)
    return { name, next }
  }
  const newStarring = () =>
    action(put, {
      doing: 'star that file',
      undo: ({ name, next }) => ({
        message: `${next ? 'Starred' : 'Unstarred'} ${leafName(name)}`,
        revert: async () => {
          if (opts.read(name) !== next) return notify.error(STAR_CHANGED)
          return put(name, !next)
        },
        target: opts.target,
      }),
    })
  const each = new SvelteMap<string, ReturnType<typeof newStarring>>()
  return {
    toggle(name: string): void {
      const now = opts.read(name)
      if (now === null) return
      const starring = each.get(name) ?? newStarring()
      each.set(name, starring)
      void starring.run(name, !now)
    },
  }
}

/** A tick-the-grid selection, for a contact sheet with bulk actions.
 *
 * The gallery and the assets page run the same selection: click a checkbox
 * to toggle, shift-click to extend a range, select everything the filter
 * leaves showing, act on the lot. It lived twice, once per page, and the
 * two copies had already drifted - the assets page's Escape handler learned
 * about `alertdialog` and the gallery's did not - so it lives here instead.
 *
 * `order` is what the grid currently shows, in the order it shows it: it
 * defines what a shift-range spans and the order the bulk actions act in,
 * and it is a getter rather than a snapshot because a filter changes it
 * under the selection. */
export class Picks {
  #names = new SvelteSet<string>()
  // Where a shift-click range starts. Not $state: nothing renders it, it is
  // only ever read by the next toggle
  #anchor: string | null = null
  #order: () => string[]

  constructor(order: () => string[]) {
    this.#order = order
  }

  has(name: string): boolean {
    return this.#names.has(name)
  }

  /** How many a bulk action would touch: a tick the filter is hiding is
   * inert, not counted, until the filter brings it back into view. */
  get size(): number {
    return this.names.length
  }

  /** The selection in the order the grid shows it. */
  get names(): string[] {
    return this.#order().filter((name) => this.#names.has(name))
  }

  /** The bulk bar's Download: the picked names as one zip from `archive`
   * (the gallery's or the asset library's route). Call it once, at setup:
   * the handle's action makes a double click fetch one, and with nothing
   * picked `run` asks for nothing */
  download(archive: (names: string[]) => Promise<unknown>, noun: string) {
    const downloading = action((picked: string[]) => archive(picked), {
      doing: `download the picked ${noun}`,
    })
    return {
      get busy() {
        return downloading.busy
      },
      run: () => {
        const picked = this.names
        if (picked.length) void downloading.run(picked)
      },
    }
  }

  /** The bulk bar's Star and Unstar: `set` each pick to `next`, one batch
   * per click (the handle's action drops a double click). Failures are one
   * toast naming the files. The Undo puts each file back as `was` read it
   * before, not to `!next` - a mixed selection has files on both sides -
   * and only the files that were set and whose star `was` still reads as
   * this set it (another writer's star is left), and only in the place it
   * was made (`target`). With nothing set, no Undo is offered. */
  star(
    set: (name: string, next: boolean) => Promise<unknown>,
    noun: string,
    opts: {
      was?: (name: string) => boolean
      /** Told after each write lands, so a page can hold it as its own */
      applied?: (name: string, next: boolean) => void
      target?: UndoTarget
    } = {},
  ) {
    const was = opts.was ?? (() => false)
    const put = async (name: string, next: boolean) => {
      await set(name, next)
      opts.applied?.(name, next)
    }
    const files = (n: number) => `${noun}${n === 1 ? '' : 's'}`
    const starring = action(
      async (picked: string[], next: boolean) => {
        const before = Object.fromEntries(
          picked.map((name) => [name, was(name)]),
        )
        const verb = next ? 'star' : 'unstar'
        const done = await setEach(picked, () => next, put, verb, files)
        return done.length ? { done, next, before } : null
      },
      {
        doing: `${noun} star`,
        undo: ({ done, next, before }) => ({
          message: `${next ? 'Starred' : 'Unstarred'} ${done.length} ${files(done.length)}`,
          revert: async () => {
            // Only the files this set changed, and only while they still read
            // as it left them: a stale view then writes nothing it should not
            const set = done.filter((name) => before[name] !== next)
            const still = set.filter((name) => was(name) === next)
            const left = set.length - still.length
            if (left && !still.length) return notify.error(STAR_CHANGED)
            if (!still.length) return
            await setEach(still, (name) => before[name], put, 'put back', files)
            if (left) {
              const m = `${left} ${files(left)}`
              const k = `${still.length} ${files(still.length)}`
              const verb =
                left === 1 ? 'was left as it is' : 'were left as they are'
              notify.success(`Undid ${k}; ${m} changed since and ${verb}.`)
            }
          },
          target: opts.target,
        }),
      },
    )
    return {
      get busy() {
        return starring.busy
      },
      run: (next: boolean) => {
        const picked = this.names
        if (picked.length) void starring.run(picked, next)
      },
    }
  }

  toggle(name: string, shift: boolean): void {
    const order = this.#order()
    const from =
      shift && this.#anchor !== null ? order.indexOf(this.#anchor) : -1
    const to = order.indexOf(name)
    if (from !== -1 && to !== -1) {
      // A range always selects: extending a selection is what shift is for,
      // and toggling each cell would make the result depend on whatever the
      // range happened to contain
      const [low, high] = from < to ? [from, to] : [to, from]
      for (const each of order.slice(low, high + 1)) this.#names.add(each)
    } else if (this.#names.has(name)) {
      this.#names.delete(name)
    } else {
      this.#names.add(name)
    }
    this.#anchor = name
  }

  selectAll(): void {
    for (const name of this.#order()) this.#names.add(name)
  }

  clear(): void {
    this.#names.clear()
    this.#anchor = null
  }

  drop(name: string): void {
    this.#names.delete(name)
  }

  /** Forget names no longer in the listing, so something deleted elsewhere
   * cannot linger in the selection and fail every later bulk action. */
  keepOnly(present: Iterable<string>): void {
    // A plain array, not a Set: this is a one-shot lookup table over a
    // selection-sized list, not reactive state, and Svelte's own Set would
    // wire it into the reactivity graph for nothing.
    const known = [...present]
    for (const name of [...this.#names])
      if (!known.includes(name)) this.#names.delete(name)
  }

  /** Keep only what failed, so a retry needs no re-ticking and the failure
   * stays visible rather than being silently dropped - and touch nothing
   * else, so a tick the filter was hiding (never attempted) survives. */
  keepFailed(attempted: string[], failed: string[]): void {
    for (const name of attempted)
      if (!failed.includes(name)) this.#names.delete(name)
  }
}

/** A name a bulk action could not act on, and why */
export interface Failure {
  name: string
  error: unknown
}

/** Run one request per name, in order, collecting what would not go.
 *
 * Sequential rather than parallel: the order makes the log readable, and a
 * selection of hundreds walking through the browser's six connections is
 * not meaningfully slower than flooding them. Returns each name that
 * failed with its error, for `keepFailed` and `reportFailures`. */
export async function actOnEach(
  names: string[],
  act: (name: string) => Promise<unknown>,
): Promise<Failure[]> {
  const failed: Failure[] = []
  for (const name of names) {
    try {
      await act(name)
    } catch (error) {
      failed.push({ name, error })
    }
  }
  return failed
}

/** Set each of `names` to `value(name)` through `put`, with one toast
 * naming what would not go (`verb`); the names that were set */
async function setEach(
  names: string[],
  value: (name: string) => boolean,
  put: (name: string, next: boolean) => Promise<void>,
  verb: string,
  files: (n: number) => string,
): Promise<string[]> {
  const failures = await actOnEach(names, (name) => put(name, value(name)))
  reportFailures(verb, failures, names.length, files(names.length))
  const bad = failures.map((f) => f.name)
  return names.filter((name) => !bad.includes(name))
}

/** One toast for a bulk action's failures, as a single one would say it:
 * "Couldn't delete 2 of 5 files: <why the first failed>", its remedy, and
 * each name with its reason in Copy details */
export function reportFailures(
  verb: string,
  failed: Failure[],
  total: number,
  noun: string,
): void {
  if (!failed.length) return
  notify.failure(
    `${verb} ${failed.length} of ${total} ${noun}`,
    failed[0].error,
    undefined,
    failed.map((f) => `${f.name}: ${describeError(f.error).title}`).join('\n'),
  )
}
