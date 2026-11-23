// What the job page's Results panel reads about a run's outputs: the
// generation metadata embedded in each image and the summary of each .otioz
// timeline, keyed by file - per file rather than per step, since each image
// of a batch carries its own seed - and the run's inputs, which are the same
// whichever output asks. Images and timelines only: the metadata route
// decodes an audio or video file whole to probe it, and nothing it would
// report is shown. A run that wrote neither (video and audio only, as H3
// and LTX do) asks its first output once, for the inputs alone.
import { api } from '../api'
import type { GalleryInput, GalleryMetadata } from '../types'

type Fields = Record<string, unknown> | null

export interface RunMetadata {
  meta: Record<string, Fields>
  media: Record<string, Fields>
  inputs: GalleryInput[]
}

export const isBundle = (file: string) => file.toLowerCase().endsWith('.otioz')

/** The files of `files` the panel asks about, by what `kindOf` says each is */
function lookups(files: string[], kindOf: (file: string) => string | null) {
  const asked = files.filter(
    (file) => kindOf(file) === 'image' || isBundle(file),
  )
  return { asked, probe: asked.length ? [] : files.slice(0, 1) }
}

/** A reader for one job's outputs. A file is asked once: a read still in
 * flight or answered is reused by the next call, and only a failed one is
 * asked again - so a call for the run's files as they grow, or after the
 * gallery moved, asks only what is new or failed */
export function runMetadataReader() {
  // Not reactive: what has been asked is bookkeeping, never shown
  const asked = new Map<string, Promise<GalleryMetadata | null>>()
  const ask = (file: string, workspace: string) => {
    const held = asked.get(file)
    if (held) return held
    const reply = api.galleryMetadata(file, workspace).catch(() => {
      asked.delete(file)
      return null
    })
    asked.set(file, reply)
    return reply
  }
  return async function read(
    files: string[],
    kindOf: (file: string) => string | null,
    workspace: string,
  ): Promise<RunMetadata> {
    const { asked: wanted, probe } = lookups(files, kindOf)
    const replies = await Promise.all(
      [...wanted, ...probe].map((file) => ask(file, workspace)),
    )
    const found: RunMetadata = { meta: {}, media: {}, inputs: [] }
    wanted.forEach((file, i) => {
      found.meta[file] = (replies[i]?.metadata as Fields) ?? null
      if (isBundle(file))
        found.media[file] = (replies[i]?.media as Fields) ?? null
    })
    found.inputs = replies.find((r) => r?.inputs?.length)?.inputs ?? []
    return found
  }
}
