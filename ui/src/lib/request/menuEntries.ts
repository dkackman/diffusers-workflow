// What a row's "more actions" menu offers, by the item's state: the same
// list whatever draws it, so it can be tested without a popover.
import type { LedgerRequest } from '../types'
import type { ItemState } from './itemState'
import type { Direction } from './reorderMath'

export type MenuEntry =
  | {
      kind: 'action'
      id: string
      label: string
      danger?: boolean
      disabled?: boolean
    }
  | { kind: 'link'; id: string; label: string; href: string }
  | { kind: 'rule' }

export const MOVES: { id: Direction; label: string }[] = [
  { id: 'up', label: 'Move up' },
  { id: 'down', label: 'Move down' },
  { id: 'top', label: 'Move to top' },
  { id: 'bottom', label: 'Move to bottom' },
]

export type MenuContext = {
  /** How many live items the plan has */
  total: number
  /** The Jobs and Gallery pages narrowed to this item */
  jobsHref: string
  outputsHref: string
  /** Whether this page can queue jobs (the run dialog is there) */
  canRun: boolean
}

const action = (
  id: string,
  label: string,
  more: { danger?: boolean; disabled?: boolean } = {},
): MenuEntry => ({ kind: 'action', id, label, ...more })

function links(row: ItemState, context: MenuContext): MenuEntry[] {
  const { attempts } = row.item
  const entries: MenuEntry[] = []
  if (attempts.length)
    entries.push({
      kind: 'link',
      id: 'jobs',
      label: 'Its jobs',
      href: context.jobsHref,
    })
  if (attempts.some((a) => a.status === 'succeeded'))
    entries.push({
      kind: 'link',
      id: 'outputs',
      label: 'Its outputs',
      href: context.outputsHref,
    })
  return entries
}

function moves(row: ItemState, total: number): MenuEntry[] {
  const first = row.position <= 1
  const last = row.position >= total
  return MOVES.map(({ id, label }) =>
    action(id, label, {
      disabled: id === 'up' || id === 'top' ? first : last,
    }),
  )
}

/** The menu's groups for a row, a rule between each: another take, where
 * its jobs and outputs are, the four moves, the note, the drop. A dropped
 * item keeps only the links - Restore is the row's own button. */
export function itemMenu(row: ItemState, context: MenuContext): MenuEntry[] {
  const groups: MenuEntry[][] = []
  const live = !row.item.dropped
  if (live && context.canRun && row.canRetake)
    groups.push([action('retake', 'Run another take…')])
  groups.push(links(row, context))
  if (live) {
    groups.push(moves(row, context.total))
    groups.push([
      action(
        'note',
        row.item.comment ? 'Edit the note' : 'Leave a note for the agent',
      ),
    ])
    groups.push([action('drop', 'Drop from plan', { danger: true })])
  }
  return groups
    .filter((group) => group.length)
    .flatMap((group, i) =>
      i ? [{ kind: 'rule' } as MenuEntry, ...group] : group,
    )
}

/** The request's own menu, in its header: an open request can be closed
 * either way, a closed one reopened. */
export function requestMenu(state: LedgerRequest['state']): MenuEntry[] {
  if (state === 'open')
    return [action('done', 'Mark done'), action('abandoned', 'Abandon')]
  return [action('open', 'Reopen')]
}
