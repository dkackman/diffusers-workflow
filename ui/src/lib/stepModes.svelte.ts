// How much of each step the workflow editor shows, keyed by step name and
// remembered per workflow. A step it has no word on opens compact; a step
// just added opens full without that being remembered, so reopening the
// file shows it the way every other step opens.
import { storageGet, storageSet } from './storage'

export type StepMode = 'collapsed' | 'compact' | 'full'

export class StepModes {
  #modes = $state<Record<string, StepMode>>({})
  #key: () => string

  /** `key` names the workflow's stored modes; read on every write, so a
   * rename of the workflow being edited moves where they are kept. */
  constructor(key: () => string) {
    this.#key = key
  }

  of(step: Record<string, any>): StepMode {
    return this.#modes[step.name] ?? 'compact'
  }

  set(step: Record<string, any>, mode: StepMode) {
    this.#modes[step.name] = mode
    this.#persist()
  }

  /** Set a step's mode for this session only. */
  mark(name: string, mode: StepMode) {
    this.#modes[name] = mode
  }

  setAll(steps: Record<string, any>[], mode: StepMode) {
    for (const step of steps) this.#modes[step.name] = mode
    this.#persist()
  }

  /** Take up the modes stored for the workflow now being edited. */
  restore() {
    this.#modes = storageGet(this.#key(), {})
  }

  reset(modes: Record<string, StepMode>) {
    this.#modes = modes
  }

  #persist() {
    storageSet(this.#key(), $state.snapshot(this.#modes))
  }
}
