import { describe, expect, it } from 'vitest'
import {
  assetMedia,
  folderOf,
  hasPicture,
  kindGlyph,
  outputMedia,
} from './media'
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

  it("builds an asset's thumbnail in the listing's workspace", () => {
    const asset = { name: 'a.png', kind: 'image', url: '/a' } as AssetFile
    const media = assetMedia(asset, 'studio')
    expect(media.thumb).toContain('workspace=studio')
    expect(media.workspace).toBe('studio')
  })
})

describe('folderOf', () => {
  // dw/runs.py split_run_path's cases: the identity before the run id
  it('is the directory before the run id, wherever the run id sits', () => {
    expect(folderOf('ltx2/Gyre/20260905-181530-a1b2c3d4/final/still.png')).toBe(
      'ltx2/Gyre',
    )
    expect(folderOf('wf/20260905-181530-a1b2c3d4/a.png')).toBe('wf')
    expect(folderOf('wf/20260905-181530-a1b2c3d4-2/a.png')).toBe('wf')
  })
  it('is the whole directory in the flat layout, and empty at the root', () => {
    expect(folderOf('workflows/flux/a.png')).toBe('workflows/flux')
    expect(folderOf('a.png')).toBe('')
  })
  it('counts only the first run id, and only a run-id-shaped segment', () => {
    expect(
      folderOf('wf/20260905-181530-a1b2c3d4/20260905-181531-b1b2c3d4/a.png'),
    ).toBe('wf')
    expect(folderOf('wf/r1/a.png')).toBe('wf/r1')
  })
})
