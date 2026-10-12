import { assetThumbnailUrl, galleryThumbnailUrl } from './apiUrls'
import { assetLibrary, ensureAssetLibrary } from './assetlib.svelte'
import { ensureOutputLibrary, outputLibrary } from './outputlib.svelte'
import { promptLibrary } from './promptlib.svelte'
import { ASSET, OUTPUT, PROMPT, reference, referenceName } from './references'
import { leafName } from './names'
import { formatMtime } from './format'
import type { AssetFile, GalleryFile } from './types'
import { matchesWords, type SuggestItem } from './ui/suggest'

// How much of a prompt's text a row shows: enough to recognize it by
const DETAIL_CHARS = 90

const BADGES: Partial<Record<AssetFile['origin'], string>> = {
  common: 'shared',
  examples: 'example',
}
// What stands in for a picture where the server makes no thumbnail
const GLYPHS: Record<string, string> = { audio: '♪', lut: '¶', text: '¶' }

/** Whether the server renders a thumbnail of `asset`: an image, or a
 * video's first frame. */
export function hasThumbnail(file: { kind: string }): boolean {
  return file.kind === 'image' || file.kind === 'video'
}

/** The first line of a prompt's text, cut to a row's width. */
export function firstLine(text: string): string {
  const line = text.trim().split('\n', 1)[0] ?? ''
  return line.length > DETAIL_CHARS
    ? `${line.slice(0, DETAIL_CHARS - 1)}…`
    : line
}

/** Where a name matched decides its place: the start of the name, then the
 * start of its file name, then anywhere. */
function rank(name: string, query: string): number {
  const q = query.trim().toLowerCase()
  if (!q) return 0
  const lower = name.toLowerCase()
  if (lower.startsWith(q)) return 0
  if (leafName(lower).startsWith(q)) return 1
  return 2
}

/** By where the name matched, then by name - or, with `keepOrder`, then
 * in the order given (outputs come newest first, which is the order that
 * matters for them). */
function ranked<T extends { name: string }>(
  entries: T[],
  query: string,
  keepOrder = false,
): T[] {
  return entries
    .map((entry, index) => ({ entry, index, rank: rank(entry.name, query) }))
    .sort(
      (a, b) =>
        a.rank - b.rank ||
        (keepOrder
          ? a.index - b.index
          : a.entry.name.localeCompare(b.entry.name)),
    )
    .map(({ entry }) => entry)
}

/** The stored prompts matching `query` - by name, text, tags or
 * description - as rows whose value is the bare name. */
export function promptChoices(query: string): SuggestItem[] {
  const names = promptLibrary.names ?? []
  const found = names
    .filter((name) =>
      matchesWords(
        query,
        name,
        promptLibrary.texts[name] ?? '',
        promptLibrary.descriptions[name] ?? '',
        ...(promptLibrary.tags[name] ?? []),
      ),
    )
    .map((name) => ({ name }))
  return ranked(found, query).map(({ name }) => ({
    value: name,
    label: name,
    detail: firstLine(promptLibrary.texts[name] ?? '') || undefined,
  }))
}

/** The assets matching `query`, optionally only one kind, as rows whose
 * value is the bare name. */
export function assetChoices(
  query: string,
  kind?: AssetFile['kind'],
): SuggestItem[] {
  ensureAssetLibrary()
  const found = (assetLibrary.items ?? []).filter(
    (asset) =>
      (!kind || asset.kind === kind) && matchesWords(query, asset.name),
  )
  return ranked(found, query).map((asset) => ({
    value: asset.name,
    label: asset.name,
    thumb: hasThumbnail(asset) ? assetThumbnailUrl(asset.name) : undefined,
    glyph: GLYPHS[asset.kind],
    badge: BADGES[asset.origin],
  }))
}

/** A run's file as a row: the workflow and file name, which run it was
 * (the gallery's v4) and when. */
export function outputRow(file: GalleryFile, value = file.name): SuggestItem {
  return {
    value,
    label: `${file.folder}/${leafName(file.name)}`,
    detail: [file.subfolder, formatMtime(file.mtime)]
      .filter(Boolean)
      .join(' · '),
    thumb: hasThumbnail(file) ? galleryThumbnailUrl(file.name) : undefined,
    glyph: GLYPHS[file.kind],
    badge: file.version ? `v${file.version}` : undefined,
  }
}

/** The files earlier runs wrote that match `query`, newest first,
 * optionally only one kind, as rows whose value is the gallery name. */
export function outputChoices(
  query: string,
  kind?: GalleryFile['kind'],
): SuggestItem[] {
  ensureOutputLibrary()
  const found = (outputLibrary.items ?? []).filter(
    (file) =>
      (!kind || file.kind === kind) &&
      matchesWords(query, file.name, file.label),
  )
  return ranked(found, query, true).map((file) => outputRow(file))
}

/** What a field offers once its text starts a reference it can look up:
 * `asset:` the asset library (thumbnails, which library), `output:` what
 * earlier runs wrote (newest first, which run), `prompt:` the
 * prompt library (searched through the text and tags too). null for
 * anything else, so the field's own suggestions stand. `kind` keeps an
 * `asset:` completion to the media a field takes. */
export function referenceCompletions(
  text: string,
  kind?: AssetFile['kind'],
): SuggestItem[] | null {
  const assetQuery = referenceName(text, ASSET)
  if (assetQuery !== null) {
    return assetChoices(assetQuery, kind).map((item) => ({
      ...item,
      value: reference(ASSET, item.value),
    }))
  }
  const outputQuery = referenceName(text, OUTPUT)
  if (outputQuery !== null) {
    return outputChoices(outputQuery, kind === 'lut' ? undefined : kind).map(
      (item) => ({ ...item, value: reference(OUTPUT, item.value) }),
    )
  }
  const promptQuery = referenceName(text, PROMPT)
  if (promptQuery !== null) {
    return promptChoices(promptQuery).map((item) => ({
      ...item,
      value: reference(PROMPT, item.value),
    }))
  }
  return null
}
