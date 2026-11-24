// What the request page says about items held for the user's review: the
// header's banner, a held row's chip, and the live region's sentence. The
// state is always the server's `review.state`; the seconds left come from
// the page's countdown (review.svelte.ts), anchored to the read.
import type { ItemState } from './itemState'
import type { LedgerRequest, LedgerReview } from '../types'

export type HoldState = LedgerReview['state']

/** The banner speaks of the most urgent state present */
const URGENCY: HoldState[] = ['due', 'holding', 'redirected']

/** The rows the banner speaks of: those in its one state */
export type HoldGroup = {
  state: HoldState
  rows: ItemState[]
  /** The row whose deadline comes first: the time the banner names */
  first: ItemState
  /** The agent's estimates summed, null when none of the rows has one */
  minutes: number | null
}

const deadline = (row: ItemState) => Date.parse(row.review?.until ?? '')

/** The group the banner speaks of, or null when nothing is held */
export function holdGroup(held: readonly ItemState[]): HoldGroup | null {
  for (const state of URGENCY) {
    const rows = held.filter((row) => row.review?.state === state)
    if (!rows.length) continue
    const first = rows.reduce((a, b) => (deadline(b) < deadline(a) ? b : a))
    const estimates = rows
      .map((row) => row.item.estimate_minutes)
      .filter((m): m is number => m != null)
    const minutes = estimates.length ? estimates.reduce((a, b) => a + b) : null
    return { state, rows, first, minutes }
  }
  return null
}

/** A deadline in the browser's locale time, hours and minutes */
export const clockTime = (until: string) =>
  new Date(until).toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
  })

const pad = (n: number) => String(n).padStart(2, '0')

/** Seconds as MM:SS, or H:MM:SS past an hour */
export function countdown(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds))
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const rest = `${pad(minutes)}:${pad(total % 60)}`
  return hours ? `${hours}:${rest}` : rest
}

const counted = (n: number) => (n === 1 ? '1 item' : `${n} items`)

const REDIRECTED_LEAD =
  "You left a note after these were held, so they won't start on their own"
const REDIRECTED_NEWS =
  "The held items won't start on their own: you left a note."

/** The banner's words. `left` is the seconds to the first deadline */
export function bannerText(
  group: HoldGroup,
  listening: boolean,
  left: number,
): string {
  const them = group.rows.length === 1 ? 'it' : 'them'
  const at = clockTime(group.first.review?.until ?? '')
  if (group.state === 'redirected')
    return listening
      ? `${REDIRECTED_LEAD}. The agent will read it and may revise them; run them here if they're right as they are.`
      : `${REDIRECTED_LEAD}, and the agent isn't connected. Run them here, or wait for the agent.`
  if (group.state === 'due')
    return listening
      ? `Review time is up: the agent should be starting ${them}.`
      : "Review time passed while the agent wasn't connected: nothing ran."
  const lead = `${counted(group.rows.length)} waiting for your OK`
  if (!listening)
    return `${lead}. The agent isn't connected, so nothing runs at ${at} unless it reconnects. Run ${them} here, or leave a note.`
  const estimate =
    group.minutes == null
      ? ''
      : ` (about ${Math.max(1, Math.round(group.minutes))} min by the agent's estimate)`
  return `${lead}${estimate}. The agent will run ${them} at ${at} if it's connected (in ${countdown(left)}), unless you leave a note.`
}

/** The rows "Run now…" would open the run dialog for: the ready ones of a
 * hold or a redirected one, or of a due one with the agent not connected to
 * run it. Empty: no button */
export function runNowKeys(group: HoldGroup, listening: boolean): string[] {
  const offered = group.state !== 'due' || !listening
  if (!offered) return []
  return group.rows
    .filter((row) => row.display === 'ready')
    .map((row) => row.item.key)
}

/** A held row's chip. `left` is its countdown's seconds (holding only) */
export function chipText(review: LedgerReview, left: number | null): string {
  if (review.state === 'redirected') return 'on hold: note left'
  if (review.state === 'due') return 'review time up'
  return `waiting for your OK · ${countdown(left ?? review.seconds_left)}`
}

/** What the live region says when the banner's state changes: never the
 * ticking digits */
export function holdNews(group: HoldGroup): string {
  if (group.state === 'redirected') return REDIRECTED_NEWS
  if (group.state === 'due') return 'Review time is up for the held items.'
  return `${counted(group.rows.length)} waiting for your OK.`
}

/** How many live items a new note would stop from starting on their own:
 * those still `holding` or `due`, not already `redirected` */
export function holdsStopped(request: LedgerRequest): number {
  return request.items.filter(
    (i) =>
      !i.dropped &&
      (i.review?.state === 'holding' || i.review?.state === 'due'),
  ).length
}
