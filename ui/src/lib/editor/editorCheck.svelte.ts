// The workflow editor's live check and where its findings lead. The check
// is the run page's RunCheck (runCheck.svelte.ts) over the document on
// screen: debounced, only the latest answer kept, quick first and with
// download sizes once the editing pauses. It runs while a file that was
// read or a new draft is open - nothing is asked while one is loading or
// failed. A step or variable a finding names is a place the editor goes to
// (findingPlace.ts).
import { onDestroy } from 'svelte'
import { api } from '../api'
import { emptyWorkflow } from '../editor'
import type { DocumentEditor } from '../editorShell.svelte'
import type { Place } from '../findingPlace'
import { RunCheck } from '../runCheck.svelte'
import type { StepModes } from '../stepModes.svelte'
import type { WorkflowDefinition } from '../types'

/** The live check over `ed`'s document. Call during component init. */
export function editorCheck(ed: DocumentEditor): RunCheck {
  const check = new RunCheck((sizes, signal) =>
    api.validate(
      { workflow: $state.snapshot(ed.doc) as WorkflowDefinition, sizes },
      signal,
    ),
  )
  // A blank draft nobody has touched has nothing to refuse yet: it waits
  // for its first edit. Validate still asks, and a draft started from an
  // example differs from the blank, so it is checked at once.
  let touched = false
  $effect(() => {
    // What was said about the last file goes with it
    if (ed.loadState !== 'loaded' && ed.loadState !== 'new') {
      check.clear()
      check.validation = null
      touched = false
      return
    }
    const doc = $state.snapshot(ed.doc)
    if (ed.loadState === 'loaded') touched = true
    else if (
      !touched &&
      JSON.stringify(doc) !== JSON.stringify(emptyWorkflow())
    )
      touched = true
    if (!touched) return
    // Keyed by the path read (a draft by 'new'), not the file name field,
    // which changes on every keystroke and is not part of what is checked
    check.changed(ed.loadedPath ?? 'new', doc)
  })
  // A verdict on the copy the server's replaced is not one on this copy;
  // the replacement's own check follows the pause a change waits for
  $effect(() => {
    if (ed.remoteLoads) check.validation = null
  })
  onDestroy(check.stop)
  return check
}

/** Where the editor takes the person: a step shown and briefly lit - the
 * flow view's click, a finding's Go to - or a variable's field focused. */
export class EditorJumps {
  /** The step lit in the list: a rail chip's hover, or a jump's for 2s */
  hovered = $state<string | null>(null)
  #ed: DocumentEditor
  #modes: StepModes

  constructor(ed: DocumentEditor, modes: StepModes) {
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
      this.#modes.set(this.#ed.doc.steps[place.index], 'full')
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
