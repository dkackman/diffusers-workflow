// The app shell's status poll (memory, health, job-finished alerts): the
// one UI timer that keeps running while the tab is hidden. The tab title
// and the job-finished alert it drives exist for the hidden tab, so it may
// never stop there (Don's ruling on #1536, Q2; ui/CLAUDE.md).
import { poll } from './poll'

export const STATUS_POLL_MS = 5000

/** Ask for the server's status now and every STATUS_POLL_MS, hidden or
 * not, until the returned function stops it */
export function pollStatus(refresh: () => unknown): () => void {
  return poll(refresh, STATUS_POLL_MS, { whileHidden: true })
}
