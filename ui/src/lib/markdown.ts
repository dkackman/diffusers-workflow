/** A small, safe Markdown subset for notes, parsed to a tree that
 * Markdown.svelte renders as Svelte nodes. Nothing here produces HTML, so
 * text that looks like markup stays text, and a link is only ever an
 * http(s)/mailto URL the browser's own URL parser accepted.
 *
 * Underscore emphasis is left out on purpose: notes quote snake_case
 * names (`cold_minutes`, `first_block`) constantly. */

export type Inline =
  | { kind: 'text'; text: string }
  | { kind: 'br' }
  | { kind: 'code'; text: string }
  | { kind: 'strong'; children: Inline[] }
  | { kind: 'em'; children: Inline[] }
  | { kind: 'link'; href: string; children: Inline[] }

export type Block =
  | { kind: 'heading'; level: 1 | 2 | 3; children: Inline[] }
  | { kind: 'paragraph'; children: Inline[] }
  | { kind: 'list'; ordered: boolean; items: Inline[][] }
  | { kind: 'code'; text: string }

const SAFE_PROTOCOLS = new Set(['http:', 'https:', 'mailto:'])

export function safeHref(url: string): string | null {
  try {
    const parsed = new URL(url)
    return SAFE_PROTOCOLS.has(parsed.protocol) ? parsed.href : null
  } catch {
    return null
  }
}

// One alternation, tried left to right at each position: code first, so a
// `**` inside backticks stays literal
const INLINE =
  /`([^`]+)`|\*\*(.+?)\*\*|\*([^*\s][^*]*?)\*|\[([^\]]+)\]\(([^()\s]+(?:\([^()\s]*\)[^()\s]*)?)\)/g

function pushText(out: Inline[], text: string) {
  if (!text) return
  const last = out[out.length - 1]
  if (last?.kind === 'text') last.text += text
  else out.push({ kind: 'text', text })
}

export function parseInline(src: string): Inline[] {
  const out: Inline[] = []
  let at = 0
  for (const m of src.matchAll(INLINE)) {
    pushText(out, src.slice(at, m.index))
    at = m.index + m[0].length
    if (m[1] !== undefined) out.push({ kind: 'code', text: m[1] })
    else if (m[2] !== undefined)
      out.push({ kind: 'strong', children: parseInline(m[2]) })
    else if (m[3] !== undefined)
      out.push({ kind: 'em', children: parseInline(m[3]) })
    else {
      const href = safeHref(m[5])
      if (href) out.push({ kind: 'link', href, children: parseInline(m[4]) })
      else pushText(out, m[0])
    }
  }
  pushText(out, src.slice(at))
  return out
}

const HEADING = /^(#{1,3})\s+(.*)$/
const BULLET = /^\s*[-*]\s+(.*)$/
const NUMBERED = /^\s*\d+[.)]\s+(.*)$/
const FENCE = /^\s*```/

const startsBlock = (line: string) =>
  FENCE.test(line) ||
  HEADING.test(line) ||
  BULLET.test(line) ||
  NUMBERED.test(line)

export function parseMarkdown(src: string): Block[] {
  const lines = src.replace(/\r\n?/g, '\n').split('\n')
  const blocks: Block[] = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (!line.trim()) {
      i++
      continue
    }
    if (FENCE.test(line)) {
      const body: string[] = []
      i++
      while (i < lines.length && !FENCE.test(lines[i])) body.push(lines[i++])
      i++ // the closing fence, or past the end when it never closed
      blocks.push({ kind: 'code', text: body.join('\n') })
      continue
    }
    const heading = HEADING.exec(line)
    if (heading) {
      blocks.push({
        kind: 'heading',
        level: heading[1].length as 1 | 2 | 3,
        children: parseInline(heading[2]),
      })
      i++
      continue
    }
    const list = BULLET.test(line)
      ? BULLET
      : NUMBERED.test(line)
        ? NUMBERED
        : null
    if (list) {
      const items: Inline[][] = []
      while (i < lines.length && list.test(lines[i]))
        items.push(parseInline(list.exec(lines[i++])![1]))
      blocks.push({ kind: 'list', ordered: list === NUMBERED, items })
      continue
    }
    // A paragraph runs until a blank line or the start of another block;
    // its single newlines are kept as line breaks - notes are written as
    // lines, not reflowed prose
    const children: Inline[] = []
    while (i < lines.length && lines[i].trim() && !startsBlock(lines[i])) {
      if (children.length) children.push({ kind: 'br' })
      children.push(...parseInline(lines[i++]))
    }
    blocks.push({ kind: 'paragraph', children })
  }
  return blocks
}
