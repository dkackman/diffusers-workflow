// @vitest-environment node
// The copy rules: what each banned term catches and spares, and that the
// glossary names every replacement. `banned_terms` in arch-metrics.mjs counts
// the matches in the source; this file pins the patterns themselves, and
// its last case is the phase gate: no copy in the source breaks them.
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { BANNED } from '../src/lib/terms'
import { measure } from './arch-metrics.mjs'

const UI = join(dirname(fileURLToPath(import.meta.url)), '..')

const SAMPLES: Record<string, { bad: string[]; good: string[] }> = {
  tilde: { bad: ['~3 min', '· ~'], good: ['about 3 min', '~/.cache'] },
  'keep-asset': {
    bad: ['Keep as asset', 'keep this file as an input asset'],
    good: ['Keep mine', 'Keep both', 'Keep editing'],
  },
  'run-noun': {
    bad: ['this run used', 'earlier runs', 'Run note'],
    good: [
      'Run it here',
      'the pipeline runs once',
      'nothing runs until you press Run',
    ],
  },
  session: { bad: ['No session is listening'], good: ['sessionStorage'] },
  starter: { bad: ['the blank starter'], good: ['workflows/templates'] },
  'star-keep': {
    bad: ['Star to keep it', 'star to keep the take'],
    good: ['Star', 'Keep starred files', 'the stars to keep'],
  },
  'latest-result': {
    bad: ['Result (latest)', 'the take: result (latest)'],
    good: ['latest take', 'the latest result'],
  },
  'held-review': {
    bad: ['held for your review', 'Held for your review.'],
    good: ['waiting for your OK', 'held for review by an agent'],
  },
}

/** `text` holds `term` whole: not inside a longer word or hyphenated name */
const hasTerm = (text: string, term: string) =>
  new RegExp(
    `(?<![\\w-])${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![\\w-])`,
  ).test(text)

const rule = (id: string) => {
  const found = BANNED.find((b) => b.id === id)
  if (!found) throw new Error(`no banned term ${id}`)
  return found
}

describe('the banned terms', () => {
  it('each rule catches what it bans and spares what it allows', () => {
    for (const [id, { bad, good }] of Object.entries(SAMPLES)) {
      const { pattern } = rule(id)
      for (const text of bad)
        expect(pattern.test(text), `${id}: ${text}`).toBe(true)
      for (const text of good)
        expect(pattern.test(text), `${id}: ${text}`).toBe(false)
    }
  })

  it('every banned term has sample rows', () => {
    expect(Object.keys(SAMPLES).sort()).toEqual(BANNED.map((b) => b.id).sort())
  })

  it("every banned term's replacement is in GLOSSARY.md, as a whole term", () => {
    const glossary = readFileSync(join(UI, 'GLOSSARY.md'), 'utf8')
    for (const { id, use } of BANNED)
      expect(hasTerm(glossary, use), `${id}: ${use}`).toBe(true)
  })

  it('a term inside a longer word is not the term', () => {
    expect(hasTerm('a jobs list', 'job')).toBe(false)
    expect(hasTerm('Starred files', 'Star')).toBe(false)
    expect(hasTerm('one job, then', 'job')).toBe(true)
    expect(hasTerm('**Star** a file', 'Star')).toBe(true)
  })
})

describe('the source', () => {
  it('no copy breaks the glossary', async () => {
    const found: string[] = (await measure()).detail.banned_terms
    const use = new Map(BANNED.map((b) => [b.id, b.use]))
    const lines = found.map(
      (at) => `${at}: use "${use.get(at.split(' ').pop()!)}"`,
    )
    expect(
      found,
      `copy that breaks ui/GLOSSARY.md:\n${lines.join('\n')}`,
    ).toEqual([])
  }, 60_000)
})
