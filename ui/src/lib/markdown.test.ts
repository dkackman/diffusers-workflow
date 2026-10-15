import { describe, expect, it } from 'vitest'
import { parseInline, parseMarkdown, safeHref } from './markdown'

describe('safeHref', () => {
  it('allows http, https and mailto only', () => {
    expect(safeHref('https://x.dev/a')).toBe('https://x.dev/a')
    expect(safeHref('mailto:a@b.c')).toBe('mailto:a@b.c')
    expect(safeHref('javascript:alert(1)')).toBeNull()
    expect(safeHref('JaVaScRiPt:alert(1)')).toBeNull()
    expect(safeHref('data:text/html,x')).toBeNull()
    expect(safeHref('/relative')).toBeNull()
  })
})

describe('parseInline', () => {
  it('reads bold, italic, code and links', () => {
    expect(parseInline('a **b** *c* `d` [e](https://x.dev)')).toEqual([
      { kind: 'text', text: 'a ' },
      { kind: 'strong', children: [{ kind: 'text', text: 'b' }] },
      { kind: 'text', text: ' ' },
      { kind: 'em', children: [{ kind: 'text', text: 'c' }] },
      { kind: 'text', text: ' ' },
      { kind: 'code', text: 'd' },
      { kind: 'text', text: ' ' },
      {
        kind: 'link',
        href: 'https://x.dev/',
        children: [{ kind: 'text', text: 'e' }],
      },
    ])
  })
  it('leaves an unsafe link as its literal text', () => {
    expect(parseInline('[x](javascript:alert(1))')).toEqual([
      { kind: 'text', text: '[x](javascript:alert(1))' },
    ])
  })
  it('leaves snake_case alone', () => {
    expect(parseInline('cold_minutes and warm_minutes')).toEqual([
      { kind: 'text', text: 'cold_minutes and warm_minutes' },
    ])
  })
  it('keeps markup-looking text as text', () => {
    expect(parseInline('<b>x</b>')).toEqual([
      { kind: 'text', text: '<b>x</b>' },
    ])
  })
})

describe('parseMarkdown', () => {
  it('reads headings, lists, code and paragraphs', () => {
    const src = [
      '## RULE',
      'Two paths, pick by target.',
      'Second line.',
      '',
      '- TV: L0 -> L7',
      '- Cinema: B3 -> P1',
      '',
      '1. one',
      '2. two',
      '',
      '```',
      'dw validate x',
      '```',
    ].join('\n')
    expect(parseMarkdown(src)).toEqual([
      { kind: 'heading', level: 2, children: [{ kind: 'text', text: 'RULE' }] },
      {
        kind: 'paragraph',
        children: [
          { kind: 'text', text: 'Two paths, pick by target.' },
          { kind: 'br' },
          { kind: 'text', text: 'Second line.' },
        ],
      },
      {
        kind: 'list',
        ordered: false,
        items: [
          [{ kind: 'text', text: 'TV: L0 -> L7' }],
          [{ kind: 'text', text: 'Cinema: B3 -> P1' }],
        ],
      },
      {
        kind: 'list',
        ordered: true,
        items: [
          [{ kind: 'text', text: 'one' }],
          [{ kind: 'text', text: 'two' }],
        ],
      },
      { kind: 'code', text: 'dw validate x' },
    ])
  })
  it('a plain note is one paragraph', () => {
    expect(parseMarkdown('just text')).toEqual([
      { kind: 'paragraph', children: [{ kind: 'text', text: 'just text' }] },
    ])
  })
  it('an unclosed fence runs to the end', () => {
    expect(parseMarkdown('```\nx')).toEqual([{ kind: 'code', text: 'x' }])
  })
})
