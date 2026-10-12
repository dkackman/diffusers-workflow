import { assetThumbnailUrl } from './apiUrls'
import { assetLibrary, ensureAssetLibrary } from './assetlib.svelte'
import { promptLibrary } from './promptlib.svelte'
import { ASSET, PROMPT, reference, referenceName } from './references'
import { leafName } from './names'
import type { AssetFile } from './types'
import { matchesWords, type SuggestItem } from './ui/suggest'

// How much of a prompt's text a row shows: enough to recognize it by
const DETAIL_CHARS = 90

const BADGES: Partial<Record<AssetFile['origin'], string>> = {
  common: 'shared',
  examples: 'example',
}
// What stands in for a picture where the server makes no thumbnail
const GLYPHS: Record<string, string> = { audio: '♪', lut: '¶' }

/** Whether the server renders a thumbnail of `asset`: an image, or a
 * video's first frame. */
export function hasThumbnail(asset: AssetFile): boolean {
  return asset.kind === 'image' || asset.kind === 'video'
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

function ranked<T extends { name: string }>(entries: T[], query: string): T[] {
  return entries
    .map((entry) => ({ entry, rank: rank(entry.name, query) }))
    .sort((a, b) => a.rank - b.rank || a.entry.name.localeCompare(b.entry.name))
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
  const found = (assetLibrary.assets ?? []).filter(
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

/** What a field offers once its text starts a reference it can look up:
 * `asset:` the asset library (thumbnails, which library), `prompt:` the
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
  const promptQuery = referenceName(text, PROMPT)
  if (promptQuery !== null) {
    return promptChoices(promptQuery).map((item) => ({
      ...item,
      value: reference(PROMPT, item.value),
    }))
  }
  return null
}
