import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./api', async () => (await import('./test/apiMock')).apiModule())
api.listAssets.mockImplementation(() => new Promise(() => {}))
api.gallery.mockImplementation(() => new Promise(() => {}))

import { assetLibrary } from './assetlib.svelte'
import { outputLibrary } from './outputlib.svelte'
import {
  earlierResults,
  fieldRows,
  kindWithArticle,
  missingOf,
  previewOf,
  pickedKind,
} from './referenceField'
import type { AssetFile, GalleryFile } from './types'
import { api } from './test/apiMock'

const workflow = {
  variables: { prompt: 'a cat' },
  steps: [
    { name: 'still', result: { content_type: 'image/png' } },
    { name: 'clip', result: { content_type: 'video/mp4' } },
    { name: 'voice', result: { content_type: 'audio/wav' } },
    { name: 'plain' },
    { name: 'here' },
  ],
}

beforeEach(() => {
  assetLibrary.workspace = 'default'
  assetLibrary.items = [
    {
      name: 'cast/priya.png',
      reference: 'asset:cast/priya.png',
      folder: 'cast',
      kind: 'image',
      size: 1,
      mtime: 1,
      origin: 'common',
      writable: true,
      url: '/inputs/cast/priya.png',
    } as AssetFile,
  ]
  outputLibrary.workspace = 'default'
  outputLibrary.items = [
    {
      name: 'shots/r1/a.png',
      folder: 'shots',
      kind: 'image',
      label: 'a.png',
      url: '/outputs/shots/r1/a.png',
      size: 2,
      mtime: 2,
    } as GalleryFile,
  ]
})

describe('earlierResults', () => {
  it('offers earlier results of the kind first', () => {
    expect(earlierResults(workflow, 4, 'audio')).toEqual([
      'previous_result:clip.audio',
      'previous_result:voice',
    ])
    expect(earlierResults(workflow, 4, 'video')).toEqual([
      'previous_result:clip',
    ])
    // Only steps before this one
    expect(earlierResults(workflow, 1, 'video')).toEqual([])
    // Any file takes every one, and nothing that is not a step's result
    expect(earlierResults(workflow, 4, 'file')).toEqual([
      'previous_result:still',
      'previous_result:clip',
      'previous_result:clip.frames',
      'previous_result:clip.audio',
      'previous_result:voice',
      'previous_result:plain',
    ])
    expect(earlierResults(workflow, 4)).toEqual(
      earlierResults(workflow, 4, 'file'),
    )
  })

  it("offers a video step's frames to an image field", () => {
    expect(earlierResults(workflow, 4, 'image')).toEqual([
      'previous_result:still',
      'previous_result:clip.frames',
    ])
  })
})

describe('fieldRows', () => {
  it('an empty field lists earlier results before the prefixes', () => {
    const rows = fieldRows('', 'image', ['previous_result:still'])!
    expect(rows[0]).toEqual({
      value: 'previous_result:still',
      detail: 'the result of step still',
    })
    expect(rows.slice(1).map((r) => r.value)).toEqual([
      'asset:',
      'output:',
      'prompt:',
    ])
  })

  it('a previous_result: start narrows the earlier results by name', () => {
    const earlier = ['previous_result:still', 'previous_result:clip.frames']
    expect(fieldRows('previous_result:cl', 'image', earlier)).toEqual([
      {
        value: 'previous_result:clip.frames',
        detail: 'the result of step clip',
      },
    ])
    expect(fieldRows('prev', 'image', earlier)?.length).toBe(2)
    // Other text is the libraries' alone
    expect(
      fieldRows('asset:pri', 'image', earlier)?.map((r) => r.value),
    ).toEqual(['asset:cast/priya.png'])
    expect(fieldRows('a dog', 'image', earlier)).toBeNull()
  })
})

describe('previewOf', () => {
  it('previews an asset, an output and an image URL', () => {
    expect(previewOf('asset:cast/priya.png', 'image')).toMatchObject({
      name: 'cast/priya.png',
      kind: 'image',
      url: '/inputs/cast/priya.png',
    })
    expect(previewOf('output:shots/r1/a.png', 'image')).toMatchObject({
      name: 'shots/r1/a.png',
      kind: 'image',
      url: '/outputs/shots/r1/a.png',
    })
    expect(previewOf('https://example.com/cat.png?x=1', 'image')).toEqual({
      name: 'https://example.com/cat.png?x=1',
      kind: 'image',
      url: 'https://example.com/cat.png?x=1',
      label: 'cat.png',
      workspace: expect.any(String),
      thumb: 'https://example.com/cat.png?x=1',
    })
    // A clip at a URL is never fetched as a picture
    expect(previewOf('/media/clip.mp4', 'video')?.thumb).toBeNull()
  })

  it('previews nothing for previous_result', () => {
    expect(previewOf('previous_result:still', 'image')).toBeNull()
    expect(previewOf('variable:image', 'image')).toBeNull()
    expect(previewOf('asset:gone.png', 'image')).toBeNull()
    expect(previewOf('output:gone.png', 'image')).toBeNull()
    expect(previewOf('cat.png', 'image')).toBeNull()
  })

  it('draws nothing for an asset or output of another kind', () => {
    expect(previewOf('asset:clips/walk.mp4', 'image')).toBeNull()
    expect(previewOf('output:shots/r1/a.png', 'video')).toBeNull()
  })

  it('names the kind of a picked file by its type, else its extension', () => {
    expect(pickedKind({ name: 'a.bin', type: 'image/png' })).toBe('image')
    expect(pickedKind({ name: 'cat.PNG', type: '' })).toBe('image')
    expect(pickedKind({ name: 'walk.mp4', type: '' })).toBe('video')
    expect(pickedKind({ name: 'song.wav', type: '' })).toBe('audio')
    expect(pickedKind({ name: 'notes', type: '' })).toBeUndefined()
  })
})

describe('missingOf', () => {
  it("stands the kind's glyph in for a library name, and nothing else", () => {
    expect(missingOf('output:shots/gone.png', 'image')).toMatchObject({
      kind: 'image',
      label: 'gone.png',
      thumb: null,
    })
    expect(missingOf('previous_result:still', 'image')).toBeNull()
    expect(missingOf('asset:gone.png', 'file')).toBeNull()
  })
})

it('names a kind with its article', () => {
  expect(kindWithArticle('image')).toBe('an image')
  expect(kindWithArticle('video')).toBe('a video')
  expect(kindWithArticle('audio')).toBe('an audio')
  expect(kindWithArticle(undefined)).toBe('a file')
})
