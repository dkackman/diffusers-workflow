// The arithmetic of moving one item of a plan: where a key press or a drop
// puts it, and the anchor that says so. A move is "put X before Y" (or at
// the end), never an index, so it still means the same thing after the
// agent has added or removed items around it.

export type Direction = 'up' | 'down' | 'top' | 'bottom'

/** The keys the focused handle answers to */
export const KEY_MOVES: Record<string, Direction> = {
  ArrowUp: 'up',
  ArrowDown: 'down',
  Home: 'top',
  End: 'bottom',
}

/** Where an item at `from` of `count` lands for a direction */
export function indexAfter(
  count: number,
  from: number,
  direction: Direction,
): number {
  if (direction === 'top') return 0
  if (direction === 'bottom') return count - 1
  const to = direction === 'up' ? from - 1 : from + 1
  return Math.min(count - 1, Math.max(0, to))
}

/** `keys` with `key` put before `before`, or last when `before` is null */
export function placeBefore(
  keys: readonly string[],
  key: string,
  before: string | null,
): string[] {
  const rest = keys.filter((k) => k !== key)
  const at = before === null ? rest.length : rest.indexOf(before)
  rest.splice(at < 0 ? rest.length : at, 0, key)
  return rest
}

/** The key `key` sits before once it is at index `to`; null for the end */
export function anchorAt(
  keys: readonly string[],
  key: string,
  to: number,
): string | null {
  return keys.filter((k) => k !== key)[to] ?? null
}

export type Move = { before: string | null; position: number; total: number }

/** The move a direction asks of `key`, with its 1-based place afterwards;
 * null when it is already there (or is not in the list). */
export function moveTarget(
  keys: readonly string[],
  key: string,
  direction: Direction,
): Move | null {
  const from = keys.indexOf(key)
  if (from < 0) return null
  const to = indexAfter(keys.length, from, direction)
  if (to === from) return null
  return {
    before: anchorAt(keys, key, to),
    position: to + 1,
    total: keys.length,
  }
}

/** The move that puts `key` at index `to` (a drop); null when it is there */
export function moveTo(
  keys: readonly string[],
  key: string,
  to: number,
): Move | null {
  const from = keys.indexOf(key)
  if (from < 0 || to === from || to < 0 || to >= keys.length) return null
  return {
    before: anchorAt(keys, key, to),
    position: to + 1,
    total: keys.length,
  }
}

/** The index a dragged row would take with the pointer at `y`: how many of
 * the other rows have their middle above it. `rows` are the rows' boxes
 * as they lay when the drag began. */
export function dropIndex(
  rows: readonly { top: number; height: number }[],
  from: number,
  y: number,
): number {
  return rows.filter((row, i) => i !== from && y > row.top + row.height / 2)
    .length
}
