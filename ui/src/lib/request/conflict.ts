// What the page says when the plan moved under an edit: what changed about
// the item being edited, who changed it, and whether the edit went on top,
// was held back to ask, or no longer means anything.
import type { LedgerHistoryEntry, LedgerItem } from '../types'
import type { Intent, Refusal } from './intents'

/** An edit the store did not write, for the page to show */
export type Conflict = {
  id: number
  /** `changed`: the item differs from what the user was looking at, so
   * the page asks. `gone`: the edit no longer applies */
  kind: 'changed' | 'gone'
  intent: Intent
  refusal: Refusal | null
  /** The item it was about; null for the request itself */
  key: string | null
  title: string
  /** The write that moved the plan, when the history still holds it */
  entry: LedgerHistoryEntry | null
  /** What changed about the item, in words */
  changes: string[]
}

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b)

/** What differs between the item as the user saw it and as it now is, on
 * the fields that make a drop or an accept a different decision. */
export function itemDiff(before: LedgerItem, after: LedgerItem): string[] {
  const changes: string[] = []
  const takes = after.attempts.length - before.attempts.length
  if (takes > 0) changes.push(`${takes} new ${takes === 1 ? 'take' : 'takes'}`)
  else if (takes < 0) changes.push('its takes changed')
  if (before.title !== after.title)
    changes.push(`it is now called ${after.title}`)
  if (before.workflow !== after.workflow) changes.push('what it runs changed')
  else if (!same(before.arguments, after.arguments))
    changes.push('its arguments changed')
  if (!same(before.depends_on, after.depends_on))
    changes.push('what it waits on changed')
  if (before.dropped !== after.dropped)
    changes.push(after.dropped ? 'it was dropped' : 'it was restored')
  return changes
}

/** A note write of the user's own (ledger.py `_noted_write`): it bumps the
 * revision outside the store, and changes nothing a plan edit is about */
const isNoteWrite = (e: LedgerHistoryEntry) =>
  e.by === 'user' && /^note (added|edited|removed)$/.test(e.why)

const between = (history: LedgerHistoryEntry[], since: number, upTo: number) =>
  history.filter((e) => e.revision > since && e.revision < upTo)

/** Whether all that moved the plan past `since` was the user's own notes */
export function onlyNotes(
  history: LedgerHistoryEntry[],
  since: number,
  upTo = Number.POSITIVE_INFINITY,
): boolean {
  const later = between(history, since, upTo)
  return later.length > 0 && later.every(isNoteWrite)
}

/** The write to name for a plan that moved past `since`: the agent's
 * latest, or failing that the latest of any but the user's notes. */
export function culprit(
  history: LedgerHistoryEntry[],
  since: number,
  upTo = Number.POSITIVE_INFINITY,
): LedgerHistoryEntry | null {
  const later = between(history, since, upTo).filter((e) => !isNoteWrite(e))
  return later.findLast((e) => e.by === 'agent') ?? later.at(-1) ?? null
}

/** The write to name for a state that moved on (`moved_on`): the newest
 * whose diff names the request's state - the one that set it as it is -
 * whoever wrote later. Only a user write closes a request, so an agent
 * write here is a reopen. Null when the history no longer holds it,
 * rather than a write that did not move it */
export function stateCulprit(
  history: LedgerHistoryEntry[],
): LedgerHistoryEntry | null {
  return history.findLast((e) => e.state != null) ?? null
}

const who = (entry: LedgerHistoryEntry | null) =>
  entry?.by === 'agent' ? 'the agent' : 'another edit'
const cite = (entry: LedgerHistoryEntry | null) =>
  entry ? `r${entry.revision}: '${entry.why}'` : ''
const inRevision = (entry: LedgerHistoryEntry | null) =>
  entry ? ` in r${entry.revision} ('${entry.why}')` : ''
const aside = (entry: LedgerHistoryEntry | null) =>
  entry ? ` (${cite(entry)})` : ''

/** The line above the items after a write went on top of someone else's */
export function appliedText(entry: LedgerHistoryEntry | null): string {
  const subject =
    entry?.by === 'agent' ? 'The agent changed the plan' : 'The plan changed'
  return `${subject} while you were editing${aside(entry)}. Your change was applied on top.`
}

const DOING: Partial<Record<Intent['kind'], string>> = {
  drop: 'dropping this',
  accept: 'choosing a take of this',
}
/** The two answers to a held-back edit: do it anyway, or leave it */
export const ANSWERS: Partial<Record<Intent['kind'], [string, string]>> = {
  drop: ['Drop anyway', 'Keep it'],
  accept: ['Accept anyway', 'Leave it'],
}

/** The card on a row whose item changed under a drop or an accept */
export function changedText(conflict: Conflict): string {
  const doing = DOING[conflict.intent.kind] ?? 'editing this'
  const what = conflict.changes.length ? `: ${conflict.changes.join(', ')}` : ''
  return `While you were ${doing}, ${who(conflict.entry)} changed it${aside(conflict.entry)}${what}.`
}

/** The card for an edit that no longer applies */
export function goneText(conflict: Conflict): string {
  const { entry, title, refusal } = conflict
  switch (refusal) {
    case 'item_dropped':
      return `Not applied: ${who(entry)} dropped ${title}${inRevision(entry)}.`
    case 'attempt_gone':
      return `Not applied: that take is no longer one of ${title}'s takes${aside(entry)}.`
    case 'moved_on':
      return entry?.by === 'agent'
        ? `Undo not applied: the agent reopened ${title} since${aside(entry)}.`
        : `Undo not applied: another edit changed the state of ${title} since${aside(entry)}.`
    case 'proposal_gone':
      return `Not kept open: the proposal to close ${title} is no longer the one you answered${aside(entry)}.`
    case 'anchor_gone':
      return `Not applied: the item ${title} was moved beside is no longer in the plan${aside(entry)}.`
    default:
      return `Not applied: ${who(entry)} removed ${title}${inRevision(entry)}.`
  }
}
