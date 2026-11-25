// Where the editor takes the person from a finding or the flow view: a step
// shown and briefly lit, or a variable's field focused (findingPlace.ts names
// the place).
import type { DocumentEditor } from '../editorShell.svelte'
import type { WorkflowDefinition } from '../types'
import type { Place } from '../findingPlace'
import type { StepModes } from '../stepModes.svelte'

/** A step shown and briefly lit - the flow view's click, a finding's Go to -
 * or a variable's field focused. */
export class EditorJumps {
  /** The step lit in the list: a rail chip's hover, or a jump's for 2s */
  hovered = $state<string | null>(null)
  #ed: DocumentEditor<WorkflowDefinition>
  #modes: StepModes

  constructor(ed: DocumentEditor<WorkflowDefinition>, modes: StepModes) {
    this.#ed = ed
    this.#modes = modes
  }

  /** Land on the form with the step scrolled into place and lit, the
   * highlight the rail chips use on hover. */
  showStep = (name: string) => {
    this.#ed.setView('form')
    this.hovered = name
    requestAnimationFrame(() => {
      document
        .getElementById('step-' + name)
        ?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    })
    setTimeout(() => {
      if (this.hovered === name) this.hovered = null
    }, 2000)
  }

  /** A finding's Go to: its step opened full and shown, or its variable's
   * field focused. */
  goTo = (place: Place) => {
    if (place.kind === 'step') {
      this.#modes.set(this.#ed.doc.steps![place.index], 'full')
      this.showStep(place.name)
      // Focus goes where the finding is: the step's name field, once the
      // list has drawn it open
      requestAnimationFrame(() => {
        document
          .getElementById('step-' + place.name)
          ?.querySelector<HTMLElement>('input.name')
          ?.focus({ preventScroll: true })
      })
      return
    }
    this.#ed.setView('form')
    requestAnimationFrame(() => {
      document.getElementById('wfvar-' + place.name)?.focus()
    })
  }
}
