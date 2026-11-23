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
}

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

  it("every banned term's replacement is in GLOSSARY.md", () => {
    const glossary = readFileSync(join(UI, 'GLOSSARY.md'), 'utf8')
    for (const { id, use } of BANNED)
      expect(glossary.includes(use), `${id}: ${use}`).toBe(true)
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
