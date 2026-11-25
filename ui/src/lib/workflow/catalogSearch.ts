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

/** The words of a text: lower-cased and cut at anything that is not a letter
 * or digit, so punctuation never rides along on a word. */
const wordsOf = (text: string): string[] =>
  text.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(Boolean)

/** The words a query asks for: lower-cased, split at punctuation and
 * whitespace, with the words that carry no meaning dropped. */
export function queryWords(query: string): string[] {
  return wordsOf(query).filter((word) => !STOP_WORDS.has(word))
}

/** Every word a card says about itself, to look query words up in. */
function textOf(name: string, card: WorkflowCard | undefined): string[] {
  return wordsOf([
    name,
    card?.description,
    card?.summary,
    card?.shape,
    ...(card?.kinds ?? []),
    ...(card?.traits ?? []),
  ]
    .join(' '))
}

/** Whether `held` holds every word, each after the one before. */
function holdsInOrder(held: string[], words: string[]): boolean {
  let from = 0
  for (const word of words) {
    const at = held.indexOf(word, from)
    if (at < 0) return false
    from = at + 1
  }
  return true
}

/** How many folders deep a workflow sits: the generic one outranks a
 * same-named one inside a model's folder. */
const depthOf = (name: string) => name.split('/').length

/** The workflows that answer a query by words, best first. A word matches a
 * whole word of the text, never part of one (`art` is not `smart`). A
 * workflow's score is how many of the query's words its text holds; only the
 * highest score above zero is kept, so `flux image` lists what holds both
 * words and `video of a cat` lists the video workflows. Ties go by how much
 * of the query the name holds, then by how shallow its folder is, then by
 * name. A query with no meaningful word matches everything, in the order
 * given. */
export function catalogMatches(
  names: string[],
  details: Record<string, WorkflowCard>,
  query: string,
): string[] {
  const words = queryWords(query)
  if (!words.length) return names
  const scored = names.map((name) => {
    const text = textOf(name, details[name])
    const named = wordsOf(name)
    return {
      name,
      score: words.filter((word) => text.includes(word)).length,
      // Ties go to the workflow whose own name holds more of the words, so
      // `text to image` lists text-to-image before every workflow that merely
      // mentions both
      inName: words.filter((word) => named.includes(word)).length,
      // and to the one that holds them in the order they were asked in:
      // `text to image` is text-to-image, not image-to-text
      inOrder: holdsInOrder(named, words) ? 1 : 0,
    }
  })
  const best = Math.max(...scored.map((entry) => entry.score))
  if (best === 0) return []
  return scored
    .filter((entry) => entry.score === best)
    .sort(
      (left, right) =>
        right.inName - left.inName ||
        right.inOrder - left.inOrder ||
        depthOf(left.name) - depthOf(right.name) ||
        left.name.localeCompare(right.name),
    )
    .map((entry) => entry.name)
}
