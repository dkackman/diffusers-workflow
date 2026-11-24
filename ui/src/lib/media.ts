// A listed file as a tile and a viewer show it: one shape for an output and
// an asset, the one glyph map for a kind with no picture, and the only two
// places a tile's thumbnail URL is built. The glyphs are spelled here and
// nowhere else (scripts/design-rules.test.ts).
import { assetThumbnailUrl, galleryThumbnailUrl } from './apiUrls'
import { leafName } from './names'
import type { AssetFile, AssetKind, GalleryFile, OutputKind } from './types'

export type MediaFile = {
  name: string
  kind: OutputKind | AssetKind
  url: string
  label: string
  workspace: string
  thumb?: string | null
  duration?: number | null
  /** The file's mtime and size: what changes when it is rewritten */
  stamp?: string
  starred?: boolean
  note?: string
}

/** Whether the server renders a picture of a file: an image, or a video's
 * first frame */
export const hasPicture = (kind: string): boolean =>
  kind === 'image' || kind === 'video'

/** What stands in for a picture where there is none: ♪ for audio, ¶ for text,
 * a LUT and anything else; nothing where a picture is shown */
export function kindGlyph(kind: string): string {
  if (hasPicture(kind)) return ''
  return kind === 'audio' ? '♪' : '¶'
}

/** An output as a tile shows it; its thumbnail is in `workspace`, the
 * listing's own */
export function outputMedia(file: GalleryFile, workspace: string): MediaFile {
  return {
    name: file.name,
    kind: file.kind,
    url: file.url,
    label: file.label,
    workspace,
    thumb: hasPicture(file.kind)
      ? galleryThumbnailUrl(file.name, workspace)
      : null,
    duration: file.duration_seconds ?? null,
    stamp: `${file.mtime}:${file.size}`,
    starred: file.starred,
    note: file.note,
  }
}

/** An asset as a tile shows it */
export function assetMedia(asset: AssetFile, workspace: string): MediaFile {
  return {
    name: asset.name,
    kind: asset.kind,
    url: asset.url,
    label: leafName(asset.name),
    workspace,
    thumb: hasPicture(asset.kind) ? assetThumbnailUrl(asset.name) : null,
    duration: asset.duration_seconds ?? null,
    stamp: `${asset.mtime}:${asset.size}`,
  }
}
