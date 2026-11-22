// Every UI timer goes through here, so every one keeps the same three
// promises (the table in changes.svelte.ts):
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
}

/** Call `fn` every `ms` - at once too, unless `immediate` is false - until
 * the returned function stops it. Written to be returned from an $effect,
 * which then stops the polling when the effect reruns or its owner goes. */
export function poll(
  fn: () => unknown,
  ms: number,
  { immediate = true, whileHidden = false }: PollOptions = {},
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

  if (whileHidden || !hidden()) {
    if (immediate) fire()
    start()
  }
  const onVisibility = () => {
    if (hidden()) pause()
    else if (timer === null) {
      fire()
      start()
    }
  }
  if (!whileHidden && typeof document !== 'undefined')
    document.addEventListener('visibilitychange', onVisibility)

  return () => {
    stopped = true
    pause()
    if (!whileHidden && typeof document !== 'undefined')
      document.removeEventListener('visibilitychange', onVisibility)
  }
}

export const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms))
