import { untrack } from 'svelte'

/** How many overlays are open. A page's own Escape handler (clearing a
 * selection, closing a detail) asks this first, so the Escape that closes
 * an overlay does not also act on the page beneath it. Each wrapper holds
 * a layer while it is open. */
let open = $state(0)

// Called from a wrapper's $effect: the count is written without being read
// as a dependency, or the effect would rerun on its own write
export function holdLayer(): () => void {
  untrack(() => (open += 1))
  let held = true
  return () => {
    if (held) {
      held = false
      untrack(() => (open -= 1))
    }
  }
}

/** Whether more than `own` overlays are open: an overlay asking whether one
 * is open above itself (the media viewer) passes the layer it holds. */
export function overlayOpen(own = 0): boolean {
  return open > own
}
