// The gallery grid's order and filter, kept out of the page so it can be
// tested on its own.
import { groupNames } from '../grouping'
import type { Picks } from '../picks.svelte'
import type { GalleryFile, OutputRole } from '../types'

/** A file whose run does not say its role counts as final */
export const roleOf = (f: GalleryFile): OutputRole => f.role ?? 'final'

const runKey = (f: GalleryFile) => `${f.folder}\u0000${f.run_id}`

/** How the grid is ordered: newest runs first (the server's order), oldest
 * runs first, or by name */
export const SORTS = ['newest', 'oldest', 'name'] as const
export type GallerySort = (typeof SORTS)[number]
/** What the grid's headings are: folders, or the day a file was written */
export const GROUPINGS = ['folder', 'date'] as const
export type Grouping = (typeof GROUPINGS)[number]
/** The header's two Segmented choices */
export const SORT_OPTIONS = [
  { value: 'newest', label: 'Newest' },
  { value: 'oldest', label: 'Oldest' },
  { value: 'name', label: 'Name' },
]
export const GROUP_OPTIONS = [
  { value: 'folder', label: 'Folder' },
  { value: 'date', label: 'Date' },
]

/** What the grid is narrowed and ordered by. `prompts` is each run's
 * submitted prompt, by run id; a run whose job has left history has none */
export interface GalleryQuery {
  text: string
  roles: OutputRole[]
  starredOnly: boolean
  sort: GallerySort
  prompts: Record<string, string>
}

/** Whether every word of `text` is somewhere in the file's name, label,
 * note or its run's prompt, case aside */
function matchesText(
  f: GalleryFile,
  words: string[],
  prompts: Record<string, string>,
): boolean {
  if (!words.length) return true
  const hay = [f.name, f.label, f.note ?? '', prompts[f.run_id] ?? '']
    .join('\n')
    .toLowerCase()
  return words.every((word) => hay.includes(word))
}

/** Runs in the server's order (newest first), finals leading inside a run.
 * A run's place is the index of its first (newest) file; built back to
 * front so the first file's index is the one that sticks, and a Map keeps
 * this linear at thousands of files. The sort is stable, so the server's
 * order holds inside each role too */
function byRun(shown: GalleryFile[], oldest: boolean): GalleryFile[] {
  const runs = new Map(shown.map((f, i) => [runKey(f), i] as const).reverse())
  const place = (f: GalleryFile) => runs.get(runKey(f))!
  const rank = (f: GalleryFile) => Number(roleOf(f) === 'intermediate')
  const way = oldest ? -1 : 1
  return shown.sort((a, b) => way * (place(a) - place(b)) || rank(a) - rank(b))
}

/** The files the grid shows, in its order: those matching every word of
 * `text` (matchesText), in one of `roles`, starred when `starredOnly`,
 * sorted by `sort` */
export function gridFiles(
  files: GalleryFile[],
  query: GalleryQuery,
): GalleryFile[] {
  const { roles, starredOnly, prompts } = query
  const words = query.text.toLowerCase().split(/\s+/).filter(Boolean)
  const shown = files.filter(
    (f) =>
      roles.includes(roleOf(f)) &&
      (!starredOnly || f.starred) &&
      matchesText(f, words, prompts),
  )
  if (query.sort === 'name')
    return shown.sort((a, b) =>
      a.name < b.name ? -1 : a.name > b.name ? 1 : 0,
    )
  return byRun(shown, query.sort === 'oldest')
}

/** The date headings, newest first */
export const DATE_GROUPS = [
  'Today',
  'Yesterday',
  'This week',
  'Earlier',
] as const
export type DateGroup = (typeof DATE_GROUPS)[number]

/** Which date heading a file written at `mtime` (unix seconds) sits under,
 * by local days: today, yesterday, the seven days before yesterday, or
 * earlier */
export function dateGroup(mtime: number, now: Date): DateGroup {
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const at = mtime * 1000
  if (at >= today.getTime()) return 'Today'
  const yesterday = new Date(today)
  yesterday.setDate(today.getDate() - 1)
  if (at >= yesterday.getTime()) return 'Yesterday'
  const week = new Date(yesterday)
  week.setDate(yesterday.getDate() - 7)
  return at >= week.getTime() ? 'This week' : 'Earlier'
}

/** How the grid's headings fall: each file's heading, the headings' order,
 * and how a heading reads. Folders sort as FolderGroups always has; dates
 * run newest first, or oldest first under that sort */
export function gridGroups(
  files: GalleryFile[],
  by: Grouping,
  sort: GallerySort,
  now: Date,
): {
  groupOf: (name: string) => string
  rank?: (group: string) => number
  heading?: (group: string) => string
} {
  const byName = new Map(files.map((f) => [f.name, f]))
  if (by === 'folder')
    return { groupOf: (name) => byName.get(name)?.folder ?? '' }
  const order: readonly string[] =
    sort === 'oldest' ? [...DATE_GROUPS].reverse() : DATE_GROUPS
  return {
    groupOf: (name) => dateGroup(byName.get(name)?.mtime ?? 0, now),
    rank: (group) => order.indexOf(group),
    heading: (group) => group,
  }
}

/** The files in the order the grid shows them, heading by heading: what
 * the viewer steps through, a shift-range spans and Select all picks. A
 * group `shut` says is collapsed shows no files, so none of its are here */
export function inGroupOrder(
  files: GalleryFile[],
  groups: ReturnType<typeof gridGroups>,
  shut: (group: string) => boolean = () => false,
): GalleryFile[] {
  const byName = new Map(files.map((f) => [f.name, f]))
  const names = files.map((f) => f.name)
  return [...groupNames(names, groups.groupOf, groups.rank).entries()]
    .flatMap(([group, inGroup]) => (shut(group) ? [] : inGroup))
    .map((name) => byName.get(name)!)
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
  /** Each run's submitted prompt, by run id, for the search */
  prompts: Record<string, string>
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
