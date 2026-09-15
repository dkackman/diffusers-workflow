import { SvelteSet } from 'svelte/reactivity'

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

  /** How many are ticked, including any the filter is currently hiding -
   * which is why `names` is the list a bulk action uses. */
  get size(): number {
    return this.#names.size
  }

  /** The selection in the order the grid shows it. */
  get names(): string[] {
    return this.#order().filter((name) => this.#names.has(name))
  }

  /** How many ticked names the filter is hiding, so a bar can say that the
   * count and what Delete would touch are not the same set. */
  get hidden(): number {
    return this.#names.size - this.names.length
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
    const known = new Set(present)
    for (const name of [...this.#names])
      if (!known.has(name)) this.#names.delete(name)
  }

  /** Keep only what failed, so a retry needs no re-ticking and the failure
   * stays visible rather than being silently dropped. */
  keepFailed(failed: string[]): void {
    this.clear()
    for (const name of failed) this.#names.add(name)
  }
}

/** Run one request per name, in order, collecting what would not go.
 *
 * Sequential rather than parallel: the order makes the log readable, and a
 * selection of hundreds walking through the browser's six connections is
 * not meaningfully slower than flooding them. Returns the names that
 * failed, for `keepFailed`. */
export async function actOnEach(
  names: string[],
  act: (name: string) => Promise<unknown>,
): Promise<string[]> {
  const failed: string[] = []
  for (const name of names) {
    try {
      await act(name)
    } catch {
      failed.push(name)
    }
  }
  return failed
}

/** Whether something modal is open, so Escape belongs to it.
 *
 * A dialog answers Escape itself - the delete confirm, the token popover,
 * the keyboard help - and a page must not take its own selection or detail
 * away underneath it. `alertdialog` is in the list because that is what
 * `ConfirmDialog` actually renders. */
export function dialogOpen(): boolean {
  return !!document.querySelector('[role="dialog"], [role="alertdialog"]')
}
