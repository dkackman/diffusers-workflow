import { describe, expect, it } from 'vitest'
import { preferenceKeys, rungKey, rungsOf } from './preferences'
import type { PreferencesRead, WorkflowDefinition } from './types'

const composing = (
  paths: string[],
  variables: Record<string, unknown> = { tier: 'standard' },
): WorkflowDefinition =>
  ({
    variables,
    steps: paths.map((path, i) => ({ name: `s${i}`, workflow: { path } })),
  }) as unknown as WorkflowDefinition

describe('preferenceKeys', () => {
  it('reads a step path, and a variable default the path names', () => {
    expect(preferenceKeys(composing(['preference:text_to_image']))).toEqual([
      'text_to_image',
    ])
    expect(
      preferenceKeys(
        composing(['variable:which'], {
          which: 'preference:video',
          tier: 'best',
        }),
      ),
    ).toEqual(['video'])
  })

  it('ignores catalog paths and repeats', () => {
    expect(
      preferenceKeys(
        composing(['templates/text-to-image', 'preference:a', 'preference:a']),
      ),
    ).toEqual(['a'])
  })
})

describe('rungKey', () => {
  it('is the one key, when the workflow declares tier', () => {
    expect(rungKey(composing(['preference:a']))).toBe('a')
  })
  it('is none with two keys, no key, or no tier variable', () => {
    expect(rungKey(composing(['preference:a', 'preference:b']))).toBeNull()
    expect(rungKey(composing(['templates/text-to-image']))).toBeNull()
    expect(rungKey(composing(['preference:a'], { prompt: 'x' }))).toBeNull()
  })
})

describe('rungsOf', () => {
  const read = {
    default_tier: 'standard',
    etag: 'e',
    preferences: {
      text_to_image: {
        etag: 'k',
        source: 'shipped',
        ladder: {},
        resolutions: {
          best: {
            target: 'models/z-image',
            skipped: [{ workflow: 'models/minimax-h3-image', reason: 'mps' }],
            notes: ['will download 41.2 GB'],
          },
          draft: { target: 'templates/text-to-image', skipped: [] },
          standard: { error: 'no target fits this host' },
        },
      },
    },
  } as unknown as PreferencesRead

  it('lists the tiers in ladder order with target, skips and error', () => {
    expect(rungsOf(read, 'text_to_image')).toEqual([
      {
        tier: 'draft',
        target: 'templates/text-to-image',
        skipped: [],
        error: null,
        notes: [],
      },
      {
        tier: 'standard',
        target: null,
        skipped: [],
        error: 'no target fits this host',
        notes: [],
      },
      {
        tier: 'best',
        target: 'models/z-image',
        skipped: [{ workflow: 'models/minimax-h3-image', reason: 'mps' }],
        error: null,
        notes: ['will download 41.2 GB'],
      },
    ])
  })

  it('an unknown key has no rungs', () => {
    expect(rungsOf(read, 'video')).toEqual([])
  })
})
