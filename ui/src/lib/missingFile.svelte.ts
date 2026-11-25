// What a page says when its link's `open=` names a file its view does not
// list (mediaViewer.ts keeps the open file in the hash): the one rule the
// Gallery, Assets, the job page's Results and Compare share, rendered by
// ui/MissingFile.svelte. Call during component init:
//
//   const stray = missingFile({ open, known, names })
//   <MissingFile leaf={stray.missing} ondismiss={stray.dismiss} />
//
// - `open()` is what open= names now; `known()` says the view's listing is
//   in and final (a running job's files may still arrive, so its page says
//   false); `names()` is every name the view lists.
// - A name `open()` gives that `names()` lacks, once `known()`, is said:
//   `missing` is its leaf. It stays said when the page then clears its
//   open= (each page does, so Back and a reload do not say it again), until
//   `dismiss()`, a workspace change, a later stray in its place, or the
//   file arriving in `names()`.
// - A name that was listed while open and then left the view is never
//   said: the viewer steps on from it (MediaViewer).
// - `gone(name)` is the page saying a listed file failed to load (its read
//   came back 404): the viewer stepped to a neighbour the listing still
//   holds and the file had been deleted. It is said as a stray is, and
//   stays said while a stale listing still names it, until `dismiss()`, a
//   workspace change or a later stray in its place.
import { untrack } from 'svelte'
import { leafName } from './names'
import { workspace } from './workspaceState.svelte'

export function missingFile(opts: {
  open: () => string | null
  known: () => boolean
  names: () => readonly string[]
}): {
  readonly missing: string | null
  dismiss(): void
  gone(name: string): void
} {
  let said = $state<string | null>(null)
  // The open name last seen listed: memory, not state
  let listed: string | null = null
  // The name `gone()` said: a listing still naming it does not unsay it
  let failed: string | null = null
  let shownIn = untrack(() => workspace.current)

  $effect(() => {
    const ws = workspace.current
    untrack(() => {
      if (ws === shownIn) return
      shownIn = ws
      said = listed = failed = null
    })
  })
  $effect(() => {
    const name = opts.open()
    if (!name || !opts.known()) return
    const here = opts.names().includes(name)
    untrack(() => {
      if (here) listed = name
      else if (name !== listed) said = name
      if (said !== failed) failed = null
    })
  })
  // The file arrived: nothing is missing any more
  $effect(() => {
    if (said !== null && said !== failed && opts.names().includes(said))
      untrack(() => (said = null))
  })

  return {
    get missing() {
      return said === null ? null : leafName(said)
    },
    dismiss() {
      said = failed = null
    },
    gone(name) {
      said = failed = name
    },
  }
}
