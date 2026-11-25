// How much of each step the workflow editor shows, keyed by step name and
// remembered per workflow. A step it has no word on opens compact; a step
// just added opens full without that being remembered, so reopening the
// file shows it the way every other step opens.
import { storageGet, storageSet } from './storage'
import type { WorkflowStep } from './types'

export type StepMode = 'collapsed' | 'compact' | 'full'

export class StepModes {
  #modes = $state<Record<string, StepMode>>({})
  #key: () => string

  /** `key` names the workflow's stored modes; read on every write, so a
   * rename of the workflow being edited moves where they are kept. */
  constructor(key: () => string) {
    this.#key = key
  }

  of(step: WorkflowStep): StepMode {
    return this.#modes[step.name] ?? 'compact'
  }

  set(step: WorkflowStep, mode: StepMode) {
    this.#modes[step.name] = mode
    this.#persist()
  }

  /** Set a step's mode for this session only. */
  mark(name: string, mode: StepMode) {
    this.#modes[name] = mode
  }

  /** A renamed step keeps its mode. One the person picked stays
   * remembered; one only marked for this session stays unstored. */
  rename(from: string, to: string) {
    if (from in this.#modes) {
      this.#modes[to] = this.#modes[from]
      delete this.#modes[from]
    }
    const stored = storageGet<Record<string, StepMode>>(this.#key(), {})
    if (!(from in stored)) return
    stored[to] = stored[from]
    delete stored[from]
    storageSet(this.#key(), stored)
  }

  setAll(steps: WorkflowStep[], mode: StepMode) {
    for (const step of steps) this.#modes[step.name] = mode
    this.#persist()
  }

  /** Take up the modes stored for the workflow now being edited. */
  restore() {
    this.#modes = storageGet(this.#key(), {})
  }

  /** The steps are now `names` (the server's copy replaced them): each
   * keeps its mode by name, and one that left is forgotten, so a step of
   * that name coming back opens as any new one does. */
  retain(names: string[]) {
    const kept = (modes: Record<string, StepMode>) =>
      Object.fromEntries(
        Object.entries(modes).filter(([n]) => names.includes(n)),
      )
    this.#modes = kept(this.#modes)
    storageSet(this.#key(), kept(storageGet(this.#key(), {})))
  }

  reset(modes: Record<string, StepMode>) {
    this.#modes = modes
  }

  #persist() {
    storageSet(this.#key(), $state.snapshot(this.#modes))
  }
}
