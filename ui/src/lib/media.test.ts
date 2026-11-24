import { describe, expect, it } from 'vitest'
import { assetMedia, hasPicture, kindGlyph, outputMedia } from './media'
import type { AssetFile, GalleryFile } from './types'

const file = (kind: GalleryFile['kind']) =>
  ({
    name: 'wf/run/out.png',
    kind,
    url: '/outputs/out.png',
    label: 'out.png',
    duration_seconds: 2.5,
  }) as GalleryFile

describe('kindGlyph', () => {
  it('maps every kind to its one glyph', () => {
    expect(kindGlyph('audio')).toBe('♪')
    expect(kindGlyph('text')).toBe('¶')
    expect(kindGlyph('lut')).toBe('¶')
    expect(kindGlyph('image')).toBe('')
    expect(kindGlyph('video')).toBe('')
    expect(hasPicture('video')).toBe(true)
    expect(hasPicture('audio')).toBe(false)
  })
})

describe('outputMedia', () => {
  it("builds an output's thumbnail in the listing's workspace", () => {
    const media = outputMedia(file('image'), 'studio')
    expect(media.thumb).toContain('workspace=studio')
    expect(media.workspace).toBe('studio')
  })

  it('carries a video duration', () => {
    expect(outputMedia(file('video'), 'studio').duration).toBe(2.5)
  })

  it('an audio file has no thumbnail', () => {
    expect(outputMedia(file('audio'), 'studio').thumb).toBeNull()
  })
})

describe('assetMedia', () => {
  it('thumbnails a picture and labels by leaf name', () => {
    const asset = {
      name: 'faces/a.png',
      kind: 'image',
      url: '/assets/a.png',
    } as AssetFile
    const media = assetMedia(asset, 'studio')
    expect(media.thumb).toContain('/thumbnail')
    expect(media.label).toBe('a.png')
    expect(assetMedia({ ...asset, kind: 'lut' }, 'studio').thumb).toBeNull()
  })
})
