// When the request page asks the server again, and which answer it keeps.
import type { LedgerRequest } from '../types'

/** While one of the request's jobs is queued or running */
export const ACTIVE_MS = 5000
/** While the request is open and nothing of its own is in flight: the plan
 * can still be rewritten, or its first job queued, by someone else */
export const IDLE_MS = 20000

/** How often to refresh `request` on a timer, or null for none: a closed
 * request with no job in flight changes only when someone writes to it -
 * reopens it, adds a note - and the page hears that from the change feed
 * (changes.svelte.ts) instead, within one CHANGES_POLL_MS. */
export function refreshInterval(request: LedgerRequest | null): number | null {
  if (!request) return null
  const inFlight = request.items.some(
    (item) => item.status === 'queued' || item.status === 'running',
  )
  if (inFlight) return ACTIVE_MS
  return request.state === 'open' ? IDLE_MS : null
}

/** The request to hold after a refresh answered with `incoming`. A refresh
 * sent before a save and answered after it carries the revision the save
 * replaced, and must not put it back; one at the same revision still
 * brings the items' newer derived state. Revisions count per request, so
 * another request's is never older or newer: `incoming` replaces it. */
export function newerOf(
  held: LedgerRequest | null | undefined,
  incoming: LedgerRequest,
): LedgerRequest {
  const older = held?.id === incoming.id && incoming.revision < held.revision
  return older ? held : incoming
}

/** The request's items as keyed rows, for a resource's `changed`: what
 * each says, less its review's `seconds_left`, which counts down between
 * any two reads of a held item without the item changing */
export function itemRows(
  request: LedgerRequest,
): [key: string, value: unknown][] {
  return request.items.map((item) => [
    item.key,
    item.review
      ? { ...item, review: { ...item.review, seconds_left: 0 } }
      : item,
  ])
}
