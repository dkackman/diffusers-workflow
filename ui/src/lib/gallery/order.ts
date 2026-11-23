// The gallery grid's order and filter, kept out of the page so it can be
// tested on its own.
import type { Picks } from '../picks.svelte'
import type { GalleryFile, OutputRole } from '../types'

/** A file whose run does not say its role counts as final */
export const roleOf = (f: GalleryFile): OutputRole => f.role ?? 'final'

const runKey = (f: GalleryFile) => `${f.folder}\u0000${f.run_id}`

/** The files the grid shows, in its order: those whose name holds
 * `filter`, in one of `roles`, starred when `starredOnly`. Finals lead
 * inside a run only: the server's order (newest first) holds between runs,
 * so runs sit where their newest file put them. The sort is stable, so it
 * holds inside each role too */
export function gridFiles(
  files: GalleryFile[],
  filter: string,
  roles: OutputRole[],
  starredOnly: boolean,
): GalleryFile[] {
  const needle = filter.toLowerCase()
  const shown = files.filter(
    (f) =>
      f.name.toLowerCase().includes(needle) &&
      roles.includes(roleOf(f)) &&
      (!starredOnly || f.starred),
  )
  // Each run's place is the index of its first (newest) file. Built back
  // to front so the first file's index is the one that sticks; a Map keeps
  // this linear at thousands of files
  const runs = new Map(shown.map((f, i) => [runKey(f), i] as const).reverse())
  const place = (f: GalleryFile) => runs.get(runKey(f))!
  const rank = (f: GalleryFile) => Number(roleOf(f) === 'intermediate')
  return shown.sort((a, b) => place(a) - place(b) || rank(a) - rank(b))
}

/** What a listing read holds: the files, and the request filter it was
 * asked under, so a reply for another filter is told from a reload */
export interface Listing {
  files: GalleryFile[]
  key: string
}

/** Edits this page made that a listing may not carry yet: files deleted,
 * and fields changed (a star, a note) */
export interface Edits {
  gone: string[]
  patched: Record<string, Partial<GalleryFile>>
}

/** The listing with the page's edits on it, so a reply that was asked for
 * before an edit cannot undo it */
export function applyEdits(files: GalleryFile[], edits: Edits): GalleryFile[] {
  const { gone, patched } = edits
  return files
    .filter((f) => !gone.includes(f.name))
    .map((f) => (f.name in patched ? { ...f, ...patched[f.name] } : f))
}

const same = (a: unknown, b: unknown) => (a || undefined) === (b || undefined)

/** The edits a landed listing has not caught up with: a deleted file is
 * settled once it is gone, a field once the listing says the same */
export function settleEdits(incoming: GalleryFile[], edits: Edits): Edits {
  const byName = new Map(incoming.map((f) => [f.name, f]))
  const patched: Edits['patched'] = {}
  for (const [name, patch] of Object.entries(edits.patched)) {
    const file = byName.get(name)
    const open = Object.entries(patch).some(
      ([key, value]) => file && !same(file[key as keyof GalleryFile], value),
    )
    if (open) patched[name] = patch
  }
  return { gone: edits.gone.filter((name) => byName.has(name)), patched }
}

/** What is held when a listing lands: the picks stay across a reload of the
 * same listing (a file briefly missing keeps its tick), and are cut back to
 * what is listed when it is another one (a workspace, a request filter) */
export function keepPicks(
  picks: Picks,
  held: Listing | undefined,
  incoming: Listing,
): Listing {
  if (held?.key !== incoming.key)
    picks.keepOnly(incoming.files.map((f) => f.name))
  return incoming
}
