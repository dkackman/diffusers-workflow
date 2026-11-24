import type { WorkflowCard } from '../types'

const STOP_WORDS = new Set([
  'a',
  'an',
  'and',
  'for',
  'of',
  'on',
  'the',
  'to',
  'with',
])

/** The words a query asks for: lower-cased, split on whitespace, with the
 * words that carry no meaning dropped. */
export function queryWords(query: string): string[] {
  return query
    .toLowerCase()
    .split(/\s+/)
    .filter((word) => word && !STOP_WORDS.has(word))
}

/** Everything a card says about itself, lower-cased, to look words up in. */
function textOf(name: string, card: WorkflowCard | undefined): string {
  return [
    name,
    card?.description,
    card?.summary,
    card?.shape,
    ...(card?.kinds ?? []),
    ...(card?.traits ?? []),
  ]
    .join(' ')
    .toLowerCase()
}

/** How many folders deep a workflow sits: the generic one outranks a
 * same-named one inside a model's folder. */
const depthOf = (name: string) => name.split('/').length

/** The workflows that answer a query by words, best first. A workflow's score
 * is how many of the query's words its text holds; only the highest score
 * above zero is kept, so `flux image` lists what holds both words and `video
 * of a cat` lists the video workflows, ties by how much of the query the name holds, then by how shallow its folder is, then by name. A query with no
 * meaningful word matches everything, in the order given. */
export function catalogMatches(
  names: string[],
  details: Record<string, WorkflowCard>,
  query: string,
): string[] {
  const words = queryWords(query)
  if (!words.length) return names
  const scored = names.map((name) => {
    const text = textOf(name, details[name])
    const named = name.toLowerCase()
    return {
      name,
      score: words.filter((word) => text.includes(word)).length,
      // Ties go to the workflow whose own name holds more of the words, so
      // `text to image` lists text-to-image before every workflow that merely
      // mentions both
      inName: words.filter((word) => named.includes(word)).length,
    }
  })
  const best = Math.max(...scored.map((entry) => entry.score))
  if (best === 0) return []
  return scored
    .filter((entry) => entry.score === best)
    .sort(
      (left, right) =>
        right.inName - left.inName ||
        depthOf(left.name) - depthOf(right.name) ||
        left.name.localeCompare(right.name),
    )
    .map((entry) => entry.name)
}
