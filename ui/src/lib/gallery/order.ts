// The gallery grid's order and filter, kept out of the page so it can be
// tested on its own.
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
