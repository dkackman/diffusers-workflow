/** The item `step` places from `current` in `items`, or null at either end
 * (no wrap: arriving back at the first file reads as "nothing happened") or
 * when `current` is no longer listed. */
export function neighbor(
  items: readonly string[],
  current: string,
  step: 1 | -1,
): string | null {
  const at = items.indexOf(current)
  if (at === -1) return null
  return items[at + step] ?? null
}
