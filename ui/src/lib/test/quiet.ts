// For "and then nothing happens" assertions. A real sleep before
// `not.toHaveBeenCalled` passes when the unwanted effect lands later than the
// sleep; on a faked clock the wait covers every debounce, poll and retry the
// UI sets, and costs no real time.
//
//   useQuietClock()          // once, at the top of the file or a describe
//   ...
//   await quiet()
//   expect(api.deleteAsset).not.toHaveBeenCalled()
import { afterEach, beforeEach, vi } from 'vitest'

/** Longer than any timer the UI sets to follow up on an action. */
export const QUIET_MS = 10_000

/** A faked clock for each test, still moving with real time so `findBy` and
 * `waitFor` behave as they do without it. */
export function useQuietClock(): void {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
  })
  afterEach(() => {
    vi.useRealTimers()
  })
}

/** Run `QUIET_MS` of the faked clock, settling the promises in between. It
 * throws if the clock is real, so it cannot pass for having waited. */
export async function quiet(): Promise<void> {
  await vi.advanceTimersByTimeAsync(QUIET_MS)
}
