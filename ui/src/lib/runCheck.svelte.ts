// The check beside Run: validate on load and after each pause in the
// editing, with `sizes=false` so typing never asks the hub; once the form
// has sat idle a while, one more check with sizes fills in download GB.
// It only informs - Run stays enabled, and the server stays the gate.
import type { ValidationResult } from './types'

/** The quiet after the last change before checking again. */
export const CHECK_AFTER_MS = 600
/** The idle time before the one check that asks the hub for sizes. */
export const SIZES_AFTER_MS = 3000

export type Verdict = { tone: 'good' | 'warn' | 'bad'; text: string }

export class RunCheck {
  validation = $state<ValidationResult | null>(null)
  /** A check the server refused outright (an unknown name, a 400), as
   * caught: shown through ErrorLine, never as its raw message. */
  error = $state<unknown>(null)
  checking = $state(false)

  // Not state: bookkeeping for which answer is still wanted
  #ask: (sizes: boolean, signal?: AbortSignal) => Promise<ValidationResult>
  #checkTimer: ReturnType<typeof setTimeout> | undefined
  #sizesTimer: ReturnType<typeof setTimeout> | undefined
  #controller: AbortController | null = null
  /** Bumped by every change: an answer from before the latest is dropped. */
  #generation = 0
  /** The latest call applied within this generation - a quick answer that
   * lands after the sizes one does not overwrite it. */
  #appliedCall = 0
  #callCount = 0
  #checkedFor = ''

  /** `ask` validates what a Run would send now: it is called when a check
   * fires, not when it is scheduled. */
  constructor(
    ask: (sizes: boolean, signal?: AbortSignal) => Promise<ValidationResult>,
  ) {
    this.#ask = ask
  }

  /** Drop every pending check and the one in flight. */
  stop = () => {
    clearTimeout(this.#checkTimer)
    clearTimeout(this.#sizesTimer)
    this.#controller?.abort()
    this.#controller = null
  }

  /** Another workflow is loading: nothing asked about the last one is
   * wanted any more. */
  clear() {
    this.stop()
    this.#generation += 1
    this.#checkedFor = ''
    this.checking = false
  }

  /** The page's word that `name` would now run with `args`: checks at once
   * for a workflow not asked about yet, after a pause for a changed value,
   * and not at all when nothing changed. */
  changed(name: string, args: Record<string, unknown>) {
    const key = `${name}\n${JSON.stringify(args)}`
    if (key === this.#checkedFor) return
    const first = !this.#checkedFor.startsWith(`${name}\n`)
    this.#checkedFor = key
    if (first) {
      this.validation = null
      this.error = null
    }
    this.#schedule(first ? 0 : CHECK_AFTER_MS)
  }

  get verdict(): Verdict | null {
    if (this.error !== null)
      return { tone: 'bad', text: 'could not check this job' }
    if (!this.validation) return null
    if (!this.validation.valid)
      return { tone: 'bad', text: 'a job would be refused' }
    if (this.validation.warnings.length) {
      return { tone: 'warn', text: 'a job would be admitted, with warnings' }
    }
    return { tone: 'good', text: 'a job would be admitted' }
  }

  #schedule(delay: number) {
    this.stop()
    this.#generation += 1
    this.#appliedCall = 0
    this.#controller = new AbortController()
    this.#checkTimer = setTimeout(() => this.#check(false), delay)
    this.#sizesTimer = setTimeout(() => this.#check(true), SIZES_AFTER_MS)
  }

  async #check(sizes: boolean) {
    const mine = this.#generation
    const call = ++this.#callCount
    const stale = () => mine !== this.#generation || call < this.#appliedCall
    this.checking = true
    try {
      const answer = await this.#ask(sizes, this.#controller?.signal)
      if (stale()) return
      this.#appliedCall = call
      this.validation = answer
      this.error = null
    } catch (e) {
      if (stale() || isAbort(e)) return
      this.#appliedCall = call
      this.validation = null
      this.error = e
    } finally {
      if (mine === this.#generation) this.checking = false
    }
  }
}

function isAbort(e: unknown): boolean {
  return e instanceof DOMException && e.name === 'AbortError'
}
