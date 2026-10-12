/** One row a suggestion list can show richer than its value: what a pick
 * writes into the field, and what helps someone recognize it - a
 * thumbnail, a line of the text it stands for, which library it is from. */
export type SuggestItem = {
  /** What choosing it puts in the field */
  value: string
  /** The row's name; the value itself when absent */
  label?: string
  /** A second, quieter line: a prompt's first words */
  detail?: string
  /** An image to show beside the name */
  thumb?: string
  /** A character standing in for a picture: ♪ for audio */
  glyph?: string
  /** A short tag after the name: which library it came from */
  badge?: string
}

/** Whether every whitespace-separated word of `query` occurs somewhere in
 * `haystack`, ignoring case - so 'pri' finds cast/priya.png and
 * 'castle dusk' finds a prompt whose text mentions both. An empty query
 * matches everything. */
export function matchesWords(query: string, ...haystack: string[]): boolean {
  const text = haystack.join('\n').toLowerCase()
  return query
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((word) => text.includes(word))
}
