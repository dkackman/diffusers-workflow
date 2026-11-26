/** The list pages' text filter: whether every whitespace-separated word of
 * `query` is somewhere in one of `fields`, case aside. A blank query matches
 * everything, and a word never spans two fields. The Workflows catalog is a
 * ranked search instead (workflow/catalogSearch.ts). */
export function matchWords(
  query: string,
  ...fields: Array<string | null | undefined>
): boolean {
  const hay = fields.join('\n').toLowerCase()
  return query
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((word) => hay.includes(word))
}
