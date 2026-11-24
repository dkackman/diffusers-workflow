// What a screen reader is told when a refresh changes the plan: two
// consecutive reads of one request diffed into a sentence or three. The
// first read says nothing - there is no change to announce.
import type { LedgerRequest } from '../types'
import { planState, type ItemDisplay, type ItemState } from './itemState'
import { planDiff, type PlanChange } from './planDiff'
import { holdGroup, holdNews } from './reviewText'

/** More changes than this in one refresh are counted, not listed */
const MAX_SENTENCES = 3

const BECAME: Partial<Record<ItemDisplay, string>> = {
  done: 'finished',
  revised: 'was revised since its take',
  failed: 'failed',
  cancelled: 'was cancelled',
  running: 'started rendering',
  queued: 'was queued',
  ready: 'is now ready to run',
  waiting: 'is now waiting on another item',
  dropped: 'was dropped from the plan',
}

const goodTakes = (state: ItemState) =>
  state.item.attempts.filter((a) => a.status === 'succeeded').length

function itemChange(before: ItemState | undefined, now: ItemState) {
  const title = now.item.title
  if (!before) return now.item.dropped ? null : `The agent added ${title}.`
  if (before.display !== now.display) {
    const became = BECAME[now.display]
    return became ? `${title} ${became}.` : null
  }
  if (now.display === 'done' && goodTakes(now) > goodTakes(before))
    return `${title} has a new take.`
  return null
}

/** What the agent edited in an item's job, said after its rename */
const EDITED: Partial<Record<NonNullable<PlanChange['what']>, string>> = {
  arguments: 'arguments',
  workflow: 'workflow',
}

/** What changed about one item: a rename first, so what follows reads
 * under the new name; then what the agent edited (`planDiff`'s lines) */
function itemChanges(
  before: ItemState | undefined,
  now: ItemState,
  edits: PlanChange[],
) {
  const said: string[] = []
  const was = before?.item.title
  if (was !== undefined && was !== now.item.title)
    said.push(`The agent renamed ${was} to ${now.item.title}.`)
  for (const edit of edits) {
    const what = edit.kind === 'changed' && edit.what && EDITED[edit.what]
    if (what) said.push(`The agent changed the ${what} of ${now.item.title}.`)
  }
  const change = itemChange(before, now)
  if (change) said.push(change)
  return said
}

/** An item the ledger deleted (left out of a write while it had no jobs)
 * is in no state of `next`: planDiff names it, under its last title */
function deleted(next: LedgerRequest, edits: PlanChange[]) {
  const held = new Set(next.items.map((item) => item.key))
  return edits
    .filter((edit) => edit.kind === 'dropped' && !held.has(edit.key))
    .map((edit) => `${edit.title} ${BECAME.dropped}.`)
}

const STATE: Record<LedgerRequest['state'], string> = {
  open: 'The request was reopened.',
  done: 'The request was marked done.',
  abandoned: 'The request was abandoned.',
}

/** One announcement for the change from `previous` to `next`, or '' when
 * nothing a listener would act on changed (or there was no previous read). */
export function announce(
  previous: LedgerRequest | null,
  next: LedgerRequest,
): string {
  if (!previous || previous.id !== next.id) return ''
  const was = planState(previous)
  const now = planState(next)
  const before = new Map(was.items.map((state) => [state.item.key, state]))
  const edits = planDiff(previous, next)
  const said = now.items.flatMap((state) =>
    itemChanges(
      before.get(state.item.key),
      state,
      edits.filter((edit) => edit.key === state.item.key),
    ),
  )
  said.push(...deleted(next, edits))
  // The review banner's state, said once when it changes (ahead of the
  // item changes, so a busy refresh does not count it away); its
  // countdown ticks on the page, never here
  const hold = holdGroup(now.held)
  if (hold && hold.state !== holdGroup(was.held)?.state)
    said.unshift(holdNews(hold))
  if (previous.state !== next.state) said.unshift(STATE[next.state])
  if (said.length <= MAX_SENTENCES) return said.join(' ')
  const more = said.length - MAX_SENTENCES
  return [
    ...said.slice(0, MAX_SENTENCES),
    `And ${more} more ${more === 1 ? 'change' : 'changes'}.`,
  ].join(' ')
}
