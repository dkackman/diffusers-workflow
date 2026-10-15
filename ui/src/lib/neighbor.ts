/** The item `step` places from `current` in `items`, or null at either end
 * (no wrap: arriving back at the first file reads as "nothing happened").
 * When `current` is no longer listed - a filter hid it, or it was unstarred
 * with Starred on - `lastAt`, where it last stood, says where to go on
 * from: forward is whatever now fills its place, back the one before it.
 * Without one an unlisted `current` has no neighbor. */
export function neighbor(
  items: readonly string[],
  current: string,
  step: 1 | -1,
  lastAt = -1,
): string | null {
  const at = items.indexOf(current)
  if (at !== -1) return items[at + step] ?? null
  if (lastAt < 0) return null
  return items[step === 1 ? lastAt : lastAt - 1] ?? null
}
