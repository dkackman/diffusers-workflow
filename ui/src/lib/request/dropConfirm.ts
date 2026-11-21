// When dropping an item asks first, and what it says: only where the drop
// does something the row does not show.
import type { ItemState } from './itemState'

const list = (names: string[]) =>
  names.length < 2
    ? names.join('')
    : `${names.slice(0, -1).join(', ')} and ${names.at(-1)}`

/** The warnings a drop of `row` carries; none means it needs no confirm.
 * Three cases: other items depend on it (the drop takes that dependency
 * away from them), no job links to it (the server deletes it outright),
 * and it is queued or running (its job goes on). */
export function dropWarnings(row: ItemState): string[] {
  const warnings: string[] = []
  const names = row.neededBy.map((dep) => dep.title)
  if (names.length === 1)
    warnings.push(
      `${names[0]} waits on this item. Dropping it removes that dependency, so it no longer waits for it.`,
    )
  else if (names.length)
    warnings.push(
      `${list(names)} wait on this item. Dropping it removes that dependency, so they no longer wait for it.`,
    )
  if (row.item.attempts.length === 0)
    warnings.push(
      'Nothing has been run for this item, so dropping it deletes it from the plan. This page can restore it only until you leave.',
    )
  if (row.display === 'queued' || row.display === 'running')
    warnings.push('Its job keeps running. Dropping does not cancel it.')
  return warnings
}

/** What a restore with nothing remembered says it did */
export function restoredText(title: string, stripped: string[]): string {
  const base = `Restored ${title} at the end of the plan.`
  if (!stripped.length) return base
  const gone =
    stripped.length === 1
      ? `${stripped[0]}, which is no longer in the plan`
      : `${list(stripped)}, which are no longer in the plan`
  return `${base} It no longer waits on ${gone}.`
}
