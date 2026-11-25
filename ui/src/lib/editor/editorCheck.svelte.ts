// The workflow editor's live check and where its findings lead. The check
// is the run page's RunCheck (runCheck.svelte.ts) over the document on
// screen: debounced, only the latest answer kept, quick first and with
// download sizes once the editing pauses. It runs while a file that was
// read or a new draft is open - nothing is asked while one is loading or
// failed. Where a finding leads is `editorJumps.svelte.ts`.
import { onDestroy } from 'svelte'
import { api } from '../api'
import { emptyWorkflow } from '../editor'
import type { DocumentEditor } from '../editorShell.svelte'
import type { WorkflowDefinition } from '../types'
import { RunCheck } from '../runCheck.svelte'

/** The live check over `ed`'s document. Call during component init. */
export function editorCheck(ed: DocumentEditor<WorkflowDefinition>): RunCheck {
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
