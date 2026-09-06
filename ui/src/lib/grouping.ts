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
