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
  assetLibrary.assets = [
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
