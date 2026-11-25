// The one clock every relative time reads ('changed 5 min ago'), so a page
// of them ages together on one timer rather than one each. It is a `poll`
// (src/lib/poll.ts): it pauses while the tab is hidden and ticks once on
// the tab's return, so a time read after a night away is right at once.
// It runs only while something reactive reads it: the first reader starts
// the timer and the last one to go stops it.
import { createSubscriber } from 'svelte/reactivity'
import { poll } from './poll'

/** Half a minute: a relative time's smallest step is a minute */
export const CLOCK_TICK_MS = 30_000

const subscribe = createSubscriber((update) =>
  poll(update, CLOCK_TICK_MS, { immediate: false }),
)

/** Now, in unix seconds; a reader inside an effect or template re-reads it
 * on each tick */
export const now: { readonly value: number } = {
  get value() {
    subscribe()
    return Date.now() / 1000
  },
}
