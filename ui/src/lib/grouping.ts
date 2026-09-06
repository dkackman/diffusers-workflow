/**
 * How a catalog name splits into a folder group and the label shown on its card.
 *
 * The group is everything but the last segment, not just the first. Catalogs
 * nest more than one level deep - workflows/templates/minimax/dialogue-short,
 * and a gallery entry whose name is <workflow identity>/<file> where the
 * identity is itself a path - so taking only the first segment collapses every
 * one of those into a single group and hides the structure the folders carry.
 */
export const groupOf = (name: string) =>
  name.includes('/') ? name.slice(0, name.lastIndexOf('/')) : ''

/** The card label: the leaf, since the group heading already shows the rest. */
export const leafOf = (name: string) =>
  name.includes('/') ? name.slice(name.lastIndexOf('/') + 1) : name

/**
 * Names bucketed by group, one pass, groups in sorted order. A page hands in
 * its own grouper when the name alone cannot say which folder an entry belongs
 * to - the gallery's names carry a run id the server has already stripped into
 * `folder`.
 */
export const groupNames = (
  names: string[],
  grouper: (name: string) => string = groupOf,
): Map<string, string[]> => {
  const buckets = new Map<string, string[]>()
  for (const name of names) {
    const group = grouper(name)
    const bucket = buckets.get(group)
    if (bucket) bucket.push(name)
    else buckets.set(group, [name])
  }
  return new Map([...buckets.entries()].sort(([a], [b]) => a.localeCompare(b)))
}
