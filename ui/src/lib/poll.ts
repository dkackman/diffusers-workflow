// Every repeating UI timer goes through here (ESLint bans `setInterval`
// anywhere else), so every one keeps the same three promises (the table in
// changes.svelte.ts). A one-shot delay such as a debounce may be a
// `setTimeout` its owner clears:
// - while the tab is hidden it does not tick, and on the tab's return it
//   fires once at once - unless the caller opts out with `whileHidden`
//   (only the app shell's status poll does: it drives the tab title and the
//   job-finished alert, which exist for the hidden tab);
// - a tick whose previous call is still pending is skipped, so a slow
//   server never has two of the same read in flight;
// - a call that throws or rejects is caught, so a failed read never stops
//   the timer.

const hidden = () =>
  typeof document !== 'undefined' && document.visibilityState === 'hidden'

export interface PollOptions {
  /** Call at once too (the default), rather than after one interval */
  immediate?: boolean
  /** Keep ticking while the tab is hidden. Only the status poll does */
  whileHidden?: boolean
  /** Stop after the first call. It is a timeout that waits out a hidden
   * tab: on the tab's return it starts its interval over, and does not
   * fire at once, even when it is not `immediate` */
  once?: boolean
}

/** Call `fn` every `ms` - at once too, unless `immediate` is false - until
 * the returned function stops it. Written to be returned from an $effect,
 * which then stops the polling when the effect reruns or its owner goes. */
export function poll(
  fn: () => unknown,
  ms: number,
  { immediate = true, whileHidden = false, once = false }: PollOptions = {},
): () => void {
  let pending = false
  let stopped = false
  let timer: ReturnType<typeof setInterval> | null = null

  const fire = () => {
    if (pending || stopped) return
    let result: unknown
    try {
      result = fn()
    } catch {
      return
    } finally {
      if (once) stop()
    }
    if (result instanceof Promise) {
      pending = true
      result
        .catch(() => {})
        .finally(() => {
          pending = false
        })
    }
  }
  const start = () => {
    if (timer === null) timer = setInterval(fire, ms)
  }
  const pause = () => {
    if (timer !== null) clearInterval(timer)
    timer = null
  }

  const unwatch = watchTab(whileHidden, pause, () => {
    if (timer !== null) return
    if (!once) fire()
    start()
  })
  function stop() {
    stopped = true
    pause()
    unwatch()
  }
  if (whileHidden || !hidden()) {
    if (immediate) fire()
    if (!stopped) start()
  }
  return stop
}

/** Call `hide` when the tab is hidden and `show` when it is back, until the
 * returned function stops it; never, for a timer that ticks while hidden */
function watchTab(whileHidden: boolean, hide: () => void, show: () => void) {
  if (whileHidden || typeof document === 'undefined') return () => {}
  const onVisibility = () => (hidden() ? hide() : show())
  document.addEventListener('visibilitychange', onVisibility)
  return () => document.removeEventListener('visibilitychange', onVisibility)
}

export const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms))
