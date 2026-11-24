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

/** What a listing read holds: the files, the request filter it was asked
 * under, so a reply for another filter is told from a reload, and the edit
 * count when its read began (`Edits.made`) */
export interface Listing {
  files: GalleryFile[]
  key: string
  began: number
  /** The workspace the listing is of, which its thumbnails are served from */
  workspace: string
}

/** A listing's rows by file name, for the changed-row mark (resource's
 * `rows`) */
export const listingRows = (listing: Listing) =>
  listing.files.map((f) => [f.name, f] as const)

type Named = { name: string }

/** Edits a page made that a listing may not carry yet: names deleted, and
 * fields changed (a star, a note), each stamped with `made` as it stood
 * after it. A listing whose read began at or after that stamp has seen it */
export interface Edits<T extends Named = GalleryFile> {
  made: number
  /** Each deleted name and its stamp */
  gone: Record<string, number>
  patched: Record<string, { fields: Partial<T>; at: number }>
}

export const noEdits = <T extends Named = GalleryFile>(): Edits<T> => ({
  made: 0,
  gone: {},
  patched: {},
})

/** `edits` with `fields` changed on `name`, stamped as the newest edit */
export function withPatch<T extends Named>(
  edits: Edits<T>,
  name: string,
  fields: Partial<T>,
): Edits<T> {
  const at = edits.made + 1
  const held = edits.patched[name]?.fields
  return {
    ...edits,
    made: at,
    patched: {
      ...edits.patched,
      [name]: { fields: { ...held, ...fields }, at },
    },
  }
}

/** `edits` with `names` deleted, stamped as the newest edit */
export function withGone<T extends Named>(
  edits: Edits<T>,
  names: Iterable<string>,
): Edits<T> {
  const at = edits.made + 1
  const gone = { ...edits.gone }
  for (const name of names) gone[name] = at
  return { ...edits, made: at, gone }
}

/** The listing with the page's edits on it, so a reply that was asked for
 * before an edit cannot undo it */
export function applyEdits<T extends Named>(files: T[], edits: Edits<T>): T[] {
  const { gone, patched } = edits
  return files
    .filter((f) => !(f.name in gone))
    .map((f) => (f.name in patched ? { ...f, ...patched[f.name].fields } : f))
}

const same = (a: unknown, b: unknown) => (a || undefined) === (b || undefined)

/** The edits a listing whose read began at `began` has not caught up with.
 * A field edit is settled by a listing begun after it, whatever it says (a
 * change made elsewhere since is the newer truth), or by one that already
 * says the same. A deleted name is settled the same way: by a listing begun
 * after the delete (a file written under the name since, or the same name
 * in another workspace, is real), or by one that lacks it */
export function settleEdits<T extends Named>(
  incoming: T[],
  began: number,
  edits: Edits<T>,
): Edits<T> {
  const byName = new Map(incoming.map((f) => [f.name, f]))
  const patched: Edits<T>['patched'] = {}
  for (const [name, patch] of Object.entries(edits.patched)) {
    const file = byName.get(name)
    const open =
      began < patch.at &&
      Object.entries(patch.fields).some(
        ([key, value]) => file && !same(file[key as keyof T], value),
      )
    if (open) patched[name] = patch
  }
  const gone: Edits<T>['gone'] = {}
  for (const [name, at] of Object.entries(edits.gone))
    if (began < at && byName.has(name)) gone[name] = at
  return { ...edits, gone, patched }
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
