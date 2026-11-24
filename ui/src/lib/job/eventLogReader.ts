// Spaces a live log's reads. Each beat asks for a read; a read that would
// start within `gap` ms of the last one waits, and every ask made while it
// waits folds into that one trailing read. This is not a poll: with no beat
// nothing is ever scheduled.
export function createReadSpacer(gap: number, run: () => void) {
  let last = -Infinity
  let timer: ReturnType<typeof setTimeout> | undefined

  function fire() {
    timer = undefined
    last = Date.now()
    run()
  }

  return {
    request() {
      if (timer !== undefined) return
      const wait = last + gap - Date.now()
      if (wait > 0) timer = setTimeout(fire, wait)
      else fire()
    },
    // Drop a waiting read (the job ended, changed, or the page went)
    cancel() {
      clearTimeout(timer)
      timer = undefined
    },
  }
}
