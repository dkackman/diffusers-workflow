import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { AssetFile } from './types'

const listAssets = vi.hoisted(() => vi.fn())
vi.mock('./api', () => ({ api: { listAssets: () => listAssets() } }))

import { assetLibrary } from './assetlib.svelte'
import { promptLibrary } from './promptlib.svelte'
import {
  assetChoices,
  firstLine,
  promptChoices,
  referenceCompletions,
} from './refcomplete.svelte'
import { workspace } from './workspace.svelte'

const asset = (
  name: string,
  kind: AssetFile['kind'] = 'image',
  origin: AssetFile['origin'] = 'workspace',
): AssetFile => ({
  name,
  reference: `asset:${name}`,
  folder: name.includes('/') ? name.split('/')[0] : '',
  kind,
  size: 1,
  mtime: 1,
  origin,
  writable: true,
  url: `/inputs/${name}`,
})

beforeEach(() => {
  assetLibrary.items = [
    asset('cast/priya.png', 'image', 'common'),
    asset('priority.png'),
    asset('cast/theme.wav', 'audio'),
    asset('clips/intro.mp4', 'video', 'examples'),
  ]
  assetLibrary.workspace = workspace.current
  promptLibrary.names = ['scenic', 'cast/priya-intro']
  promptLibrary.texts = {
    scenic: 'A castle at dusk\nwide shot',
    'cast/priya-intro': 'Priya walks in',
  }
  promptLibrary.tags = { scenic: ['landscape'], 'cast/priya-intro': [] }
  promptLibrary.descriptions = { scenic: '', 'cast/priya-intro': 'opening' }
})
afterEach(() => listAssets.mockReset())

describe('referenceCompletions', () => {
  it('leaves a field that has not started a library reference alone', () => {
    for (const text of ['', 'a cat', 'variable:x', 'previous_result:a'])
      expect(referenceCompletions(text)).toBeNull()
  })

  it('offers assets after asset:, found by any part of the name', () => {
    const values = referenceCompletions('asset:pri')!.map((i) => i.value)
    // the name's own start first, then a file name's start
    expect(values).toEqual(['asset:priority.png', 'asset:cast/priya.png'])
  })

  it('keeps asset completions to the kind a media field takes', () => {
    expect(
      referenceCompletions('asset:', 'video')!.map((i) => i.value),
    ).toEqual(['asset:clips/intro.mp4'])
  })

  it('offers prompts after prompt:, searched through their text too', () => {
    expect(referenceCompletions('prompt:castle')!.map((i) => i.value)).toEqual([
      'prompt:scenic',
    ])
  })
})

describe('assetChoices', () => {
  it('shows an image, a glyph for other media, and the library', () => {
    const rows = assetChoices('cast')
    expect(rows.find((r) => r.value === 'cast/priya.png')).toMatchObject({
      thumb: expect.stringContaining(
        '/api/gallery/asset%3Acast/priya.png/thumbnail',
      ),
      badge: 'shared',
    })
    expect(rows.find((r) => r.value === 'cast/theme.wav')).toMatchObject({
      glyph: '♪',
      thumb: undefined,
      badge: undefined,
    })
  })

  it('loads the library on first use, once', async () => {
    assetLibrary.workspace = undefined
    listAssets.mockReturnValue(new Promise(() => {}))
    assetChoices('')
    assetChoices('x')
    await Promise.resolve()
    expect(listAssets).toHaveBeenCalledTimes(1)
  })
})

describe('promptChoices', () => {
  it('matches every word, across name, text, tags and description', () => {
    expect(promptChoices('castle dusk').map((r) => r.value)).toEqual(['scenic'])
    expect(promptChoices('landscape').map((r) => r.value)).toEqual(['scenic'])
    expect(promptChoices('opening').map((r) => r.value)).toEqual([
      'cast/priya-intro',
    ])
    expect(promptChoices('castle walks')).toEqual([])
  })

  it("shows a prompt's first line under its name", () => {
    expect(promptChoices('scenic')[0].detail).toBe('A castle at dusk')
  })
})

it('cuts a long first line to a row', () => {
  const line = firstLine('x'.repeat(200))
  expect(line.length).toBe(90)
  expect(line.endsWith('…')).toBe(true)
})

describe('output: completions', () => {
  beforeEach(async () => {
    const { outputLibrary } = await import('./outputlib.svelte')
    outputLibrary.workspace = workspace.current
    outputLibrary.items = [
      {
        name: 'portrait/20261002-000000-bbbbbbbb/final/out.png',
        folder: 'portrait',
        subfolder: 'final',
        run_id: '20261002-000000-bbbbbbbb',
        version: 2,
        url: '/outputs/x',
        kind: 'image',
        size: 1,
        mtime: 2,
        label: 'out.png',
      },
      {
        name: 'clip/20261001-000000-aaaaaaaa/shot.mp4',
        folder: 'clip',
        subfolder: '',
        run_id: '20261001-000000-aaaaaaaa',
        version: 1,
        url: '/outputs/y',
        kind: 'video',
        size: 1,
        mtime: 1,
        label: 'shot.mp4',
      },
    ]
  })

  it('offers what earlier runs wrote, newest first, with the run', () => {
    const rows = referenceCompletions('output:')!
    expect(rows.map((r) => r.value)).toEqual([
      'output:portrait/20261002-000000-bbbbbbbb/final/out.png',
      'output:clip/20261001-000000-aaaaaaaa/shot.mp4',
    ])
    expect(rows[0]).toMatchObject({ label: 'portrait/out.png', badge: 'v2' })
    expect(rows[0].detail).toContain('final')
  })

  it('keeps to a media field’s kind', () => {
    expect(
      referenceCompletions('output:', 'video')!.map((r) => r.value),
    ).toEqual(['output:clip/20261001-000000-aaaaaaaa/shot.mp4'])
  })
})
