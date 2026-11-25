// What ReferenceField offers and shows for a value: the earlier steps'
// results a field of a kind can take, its suggestion rows with those first,
// and the file a reference names as a tile draws it - or nothing, when the
// field cannot tell what the value is (a step's result, a variable, a name
// no listing holds). Never a rewrite of the value: a preview is read off it.
import { assetMedia, outputMedia, type MediaFile } from './media'
import type { MediaKind } from './mediaArguments'
import { assetLibrary, assetNamed } from './assetlib.svelte'
import { outputLibrary, outputNamed } from './outputlib.svelte'
import { referenceCompletions } from './refcomplete.svelte'
import { referenceSuggestions } from './referenceSuggestions'
import { ASSET, OUTPUT, PREVIOUS_RESULT, referenceName } from './references'
import { leafName } from './names'
import type { AssetKind, WorkflowDefinition } from './types'
import type { SuggestItem } from './ui/suggest'
import { workspace } from './workspace.svelte'

/** What a field takes: one kind of media, or any file */
export type FieldKind = MediaKind | 'file'

/** The media kind a library is filtered to; none for any file */
export function mediaKindOf(
  kind: FieldKind | undefined,
): MediaKind | undefined {
  return kind === 'file' ? undefined : kind
}

/** The kind as copy names it, with its article: "an image", "a video",
 * "an audio", "a file" */
export function kindWithArticle(
  kind: FieldKind | AssetKind | undefined,
): string {
  const word = kind ?? 'file'
  return `${/^[aeiou]/.test(word) ? 'an' : 'a'} ${word}`
}

/** The step a `previous_result:` reference names, without its `.frames`,
 * `.audio` or `@member` */
function stepOf(ref: string): string {
  return (referenceName(ref, PREVIOUS_RESULT) ?? '').split(/[.@]/, 1)[0]
}

/** Whether a step's result, or the part of it `ref` names, is `kind` */
function resultIs(ref: string, contentType: string, kind: MediaKind): boolean {
  if (ref.endsWith('.frames'))
    return kind === 'image' && contentType.startsWith('video')
  if (ref.endsWith('.audio'))
    return kind === 'audio' && contentType.startsWith('video')
  return contentType.startsWith(kind)
}

/** The `previous_result:` references a field in step `stepIndex` can take:
 * the steps before it, with a video's `.frames` for an image field and its
 * `.audio` for an audio one; every one for any file. */
export function earlierResults(
  workflow: WorkflowDefinition,
  stepIndex: number,
  kind?: FieldKind,
): string[] {
  const media = mediaKindOf(kind)
  const steps = workflow.steps ?? []
  return referenceSuggestions(workflow, stepIndex).filter((ref) => {
    if (referenceName(ref, PREVIOUS_RESULT) === null) return false
    if (!media) return true
    const step = steps.find((s) => s.name === stepOf(ref))
    return resultIs(ref, String(step?.result?.content_type ?? ''), media)
  })
}

/** Whether `text` is empty or starts a `previous_result:` reference - spelled
 * two letters in, or with the prefix whole */
function startsEarlier(text: string): boolean {
  if (text === '') return true
  const lower = text.toLowerCase()
  return (
    lower.startsWith(PREVIOUS_RESULT) ||
    (text.length >= 2 && PREVIOUS_RESULT.startsWith(lower))
  )
}

/** A field's suggestion rows for `text`: the earlier results first while
 * the field is empty or starts one, then the libraries' rows. */
export function fieldRows(
  text: string,
  kind: FieldKind | undefined,
  earlier: string[],
): SuggestItem[] | null {
  const completions = referenceCompletions(text, mediaKindOf(kind))
  if (!startsEarlier(text)) return completions
  const lower = text.toLowerCase()
  const rows: SuggestItem[] = earlier
    .filter((ref) => ref.toLowerCase().startsWith(lower))
    .map((ref) => ({ value: ref, detail: `the result of step ${stepOf(ref)}` }))
  rows.push(...(completions ?? []))
  return rows.length ? rows : null
}

/** A typed web address or server path as a tile draws it: its picture is
 * the URL itself only for an image. null for anything else. */
function urlMedia(
  value: string,
  kind: FieldKind | undefined,
): MediaFile | null {
  const media = mediaKindOf(kind)
  if (!media || !/^(https?:\/\/|\/)/.test(value)) return null
  return {
    name: value,
    kind: media,
    url: value,
    label: leafName(value.replace(/[?#].*$/, '')) || value,
    workspace: workspace.current,
    thumb: media === 'image' ? value : null,
  }
}

/** The file `value` names, as a tile draws it: a listed asset or output, or
 * a typed URL. null for anything the field cannot show - a step's result,
 * a variable, a name no listing holds. */
export function previewOf(
  value: string,
  kind: FieldKind | undefined,
): MediaFile | null {
  const media = mediaKindOf(kind)
  // A file of another kind than the field takes is not shown as its value
  const fits = (file: { kind: string } | undefined) =>
    !!file && (!media || file.kind === media)
  const assetName = referenceName(value, ASSET)
  if (assetName !== null) {
    const asset = assetNamed(assetName)
    return asset && fits(asset)
      ? assetMedia(asset, assetLibrary.workspace ?? workspace.current)
      : null
  }
  const outputName = referenceName(value, OUTPUT)
  if (outputName !== null) {
    const file = outputNamed(outputName)
    return file && fits(file)
      ? outputMedia(file, outputLibrary.workspace ?? workspace.current)
      : null
  }
  return urlMedia(value, kind)
}

const EXTENSION_KINDS: Record<string, MediaKind> = {
  '.jpg': 'image',
  '.jpeg': 'image',
  '.png': 'image',
  '.gif': 'image',
  '.bmp': 'image',
  '.webp': 'image',
  '.mp4': 'video',
  '.avi': 'video',
  '.mkv': 'video',
  '.mov': 'video',
  '.webm': 'video',
  '.wav': 'audio',
  '.mp3': 'audio',
  '.flac': 'audio',
  '.ogg': 'audio',
}

/** The media kind a file is: its MIME type's, else - when the system gave
 * none (`.mkv` on some) - its extension's. undefined when neither says. */
export function pickedKind(file: {
  name: string
  type: string
}): MediaKind | undefined {
  if (file.type) {
    const fromType = file.type.split('/', 1)[0]
    return fromType === 'image' || fromType === 'video' || fromType === 'audio'
      ? fromType
      : undefined
  }
  const dot = file.name.lastIndexOf('.')
  return dot < 0
    ? undefined
    : EXTENSION_KINDS[file.name.slice(dot).toLowerCase()]
}

/** What stands in for a library reference the listing does not hold - a
 * deleted output, an asset renamed away: the kind's glyph, never a picture.
 * null for any other value, or a field of no one kind. */
export function missingOf(
  value: string,
  kind: FieldKind | undefined,
): MediaFile | null {
  const media = mediaKindOf(kind)
  const name = referenceName(value, ASSET) ?? referenceName(value, OUTPUT)
  if (!media || !name) return null
  // A listed file of another kind is not missing: it shows nothing
  const listed =
    referenceName(value, ASSET) !== null ? assetNamed(name) : outputNamed(name)
  if (listed) return null
  return {
    name: value,
    kind: media,
    url: '',
    label: leafName(name),
    workspace: workspace.current,
    thumb: null,
  }
}
