// A listed file as a tile and a viewer show it: one shape for an output and
// an asset, the one glyph map for a kind with no picture, and the only two
// places a tile's thumbnail URL is built. The glyphs are spelled here and
// nowhere else (scripts/design-rules.test.ts).
import {
  appendQuery,
  assetThumbnailUrl,
  galleryThumbnailUrl,
  outputUrl,
} from './apiUrls'
import { distinctLeaves, leafName } from './names'
import type {
  AssetFile,
  AssetKind,
  GalleryFile,
  LedgerOutput,
  OutputKind,
} from './types'

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

/** Each file's label in a view of `names`: the tail that tells same-leaf
 * files apart (distinctLeaves), where its leaf alone would not. `key` is
 * what is compared, when part of a name is told apart another way (the
 * Gallery's run, by its version badge) */
export function viewLabels(
  names: string[],
  key: (name: string) => string = (name) => name,
): (media: MediaFile) => MediaFile {
  const tails = distinctLeaves(names.map(key))
  return (media) => {
    const tail = tails.get(key(media.name))
    return tail && tail !== leafName(media.name)
      ? { ...media, label: tail }
      : media
  }
}

/** viewLabels' key for outputs: a name without its run folder, since two
 * runs' files are told apart by their version badge and a run id is never
 * compared (the Gallery, Compare) */
export function withoutRun(files: GalleryFile[]): (name: string) => string {
  const runs = new Map(files.map((f) => [f.name, f.run_id]))
  return (name) => {
    const run = runs.get(name)
    return run ? name.replace(`/${run}/`, '/') : name
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
    thumb: hasPicture(asset.kind)
      ? assetThumbnailUrl(asset.name, workspace)
      : null,
    duration: asset.duration_seconds ?? null,
    stamp: `${asset.mtime}:${asset.size}`,
    note: asset.note,
  }
}

/** A request's job output (the ledger's name and kind) as a tile shows it,
 * in the request's workspace */
export function ledgerMedia(
  output: LedgerOutput,
  workspace: string,
): MediaFile {
  return {
    name: output.name,
    kind: output.kind,
    url: outputUrl(output.name, undefined, workspace),
    label: leafName(output.name),
    workspace,
    thumb: hasPicture(output.kind)
      ? galleryThumbnailUrl(output.name, workspace)
      : null,
  }
}

/** One of a job's outputs (its manifest's name and the kind the server
 * gave it) as a tile shows it. The job id rides on the file and on its
 * thumbnail: two runs of a workflow write the same names, and the browser
 * would show this job the picture it cached from the last one */
export function jobMedia(
  name: string,
  kind: OutputKind,
  jobId: string,
  workspace: string,
): MediaFile {
  return {
    name,
    kind,
    url: outputUrl(name, jobId, workspace),
    label: leafName(name),
    workspace,
    thumb: hasPicture(kind)
      ? appendQuery(galleryThumbnailUrl(name, workspace), 'v', jobId)
      : null,
  }
}
