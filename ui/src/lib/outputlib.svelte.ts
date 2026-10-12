import { api } from './api'
import { lazyListing } from './lazylist.svelte'
import type { GalleryFile } from './types'

/** What earlier runs wrote, as the editors' pickers share it - the gallery
 * listing, newest first. A file's gallery name is what an `output:`
 * reference names. */
const outputs = lazyListing<GalleryFile>(() =>
  api.gallery().then((result) => result.files),
)
export const outputLibrary = outputs.listing
export const ensureOutputLibrary = outputs.ensure
export const loadOutputLibrary = outputs.load

/** The file an `output:` name resolves to in the loaded listing: the one
 * it names, or for a `latest` name the newest file that name would find
 * (the listing is newest first). */
export function outputNamed(name: string): GalleryFile | undefined {
  return outputLibrary.items?.find(
    (file) => file.name === name || latestName(file) === name,
  )
}

/** The name with `latest` in the run's place: the newest run that holds
 * this file, so a later stage keeps naming the newest product without
 * being edited after every run. null for a file that no run wrote. */
export function latestName(file: GalleryFile): string | null {
  if (!file.run_id) return null
  const prefix = `${file.folder}/${file.run_id}/`
  return file.name.startsWith(prefix)
    ? `${file.folder}/latest/${file.name.slice(prefix.length)}`
    : null
}
