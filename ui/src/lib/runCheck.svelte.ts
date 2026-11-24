// The check beside Run: validate on load and after each pause in the
// editing, with `sizes=false` so typing never asks the hub; once the form
// has sat idle a while, one more check with sizes fills in download GB.
// It only informs - Run stays enabled, and the server stays the gate.
import type { PlacedFindings } from './findings'
import type { ValidationResult } from './types'

/** The quiet after the last change before checking again. */
export const CHECK_AFTER_MS = 600
/** The idle time before the one check that asks the hub for sizes. */
export const SIZES_AFTER_MS = 3000

export type Verdict = {
  tone: 'good' | 'warn' | 'bad'
  text: string
  remedy?: string
  /** How many things stand between the values and a job: the errors and a
   * bare refusal, which a Run gated on this verdict names */
  issues: number
}

const plural = (n: number, one: string, many: string) =>
  `${n} ${n === 1 ? one : many}`

/** What a validate answer says about running now, in plain words - the one
 * verdict the run check and the editor both show. It reads `valid`,
 * `errors`, `access_blocked`, `not_found` and what `placed` holds, and
 * derives nothing else. */
export function verdictOf(
  validation: ValidationResult | null,
  error: unknown,
  placed: PlacedFindings,
): Verdict | null {
  if (error !== null && error !== undefined)
    return { tone: 'bad', text: "Couldn't check this workflow.", issues: 0 }
  if (!validation) return null
  if (!validation.valid) {
    const bad = (list: { tone: string }[]) =>
      list.filter((f) => f.tone === 'bad').length
    const issues =
      bad(placed.general) +
      Object.values(placed.byVariable).reduce((n, l) => n + bad(l), 0)
    return {
      tone: 'bad',
      text: `A job would be refused: ${plural(issues, 'issue', 'issues')} to fix.`,
      issues,
    }
  }
  const downloads = validation.plan?.downloads_required ?? []
  if (downloads.some((d) => d.access_blocked)) {
    return {
      tone: 'warn',
      text: 'This model needs Hugging Face access.',
      remedy:
        'Accept its terms on Hugging Face, then sign in on the server with hf auth login.',
      issues: 0,
    }
  }
  if (downloads.some((d) => d.not_found)) {
    return {
      tone: 'warn',
      text: "A model this workflow downloads can't be found.",
      remedy: 'Check the model name in the workflow.',
      issues: 0,
    }
  }
  const noted = Object.values(placed.byVariable).filter((list) =>
    list.some((f) => f.tone === 'warn'),
  ).length
  if (noted) {
    return {
      tone: 'warn',
      text: `Ready to run. Read the notes under ${plural(noted, 'field', 'fields')}.`,
      issues: 0,
    }
  }
  return { tone: 'good', text: 'Ready to run.', issues: 0 }
}

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
