// The grid plumbing the Gallery and the Assets page share around their
// contact sheets: opening a linked file's shut group and scrolling to its tile
// once, and Escape clearing the picks. Each page keeps its own rule where they differ, passed
// in here (#1670 stage C); whether the rules should become one is Don's call.
import { tick, untrack } from 'svelte'
import type { Picks } from './picks.svelte'
import { overlayOpen } from './ui/layers.svelte'

/** The open file's tile, brought on screen. One any part of which already
 * shows stays put, so opening a tile by a click never moves the grid */
function scrollToOpenTile(): void {
  const tile = document.querySelector('.cellwrap .filetile.active')
  if (!tile) return
  const { top, bottom } = tile.getBoundingClientRect()
  if (bottom > 0 && top < window.innerHeight) return
  tile.scrollIntoView?.({ block: 'center' })
}

/** A link to a file in a shut group opens the group, so the viewer's count
 * and the grid agree, and the grid scrolls to the file's tile; once per file,
 * so the person can shut the group or scroll away again. Call
 * during component init. `place(name)` is where the open file sits (its
 * group, its library), or undefined while the page does not list it; the
 * first time a name has a place, `reveal(name, place)` runs, untracked */
export function revealOnce<T>(
  open: () => string | null,
  place: (name: string) => T | undefined,
  reveal: (name: string, place: T) => void,
): void {
  // Memory, not state: nothing renders it
  let revealed: string | null = null
  $effect(() => {
    const name = open()
    if (!name) return
    const where = place(name)
    if (where === undefined) return
    untrack(() => {
      if (name === revealed) return
      revealed = name
      reveal(name, where)
      // after the group it just opened has rendered the tile
      void tick().then(scrollToOpenTile)
    })
  })
}

/** The window's keydown for a grid: Escape clears the picks, unless an
 * overlay is open (the viewer and a dialog answer it themselves). With
 * `keepHidden`, an Escape when no shown file is picked leaves the ticks a
 * filter hides (the Gallery's rule); without it they go too (the Assets
 * page's) */
export function clearOnEscape(
  picks: Picks,
  opts: { keepHidden?: boolean } = {},
): (e: KeyboardEvent) => void {
  return (e) => {
    if (e.key !== 'Escape' || overlayOpen()) return
    if (!opts.keepHidden || picks.size) picks.clear()
  }
}
