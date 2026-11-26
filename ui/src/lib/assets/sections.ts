// The Assets page's filter, order and library sections, kept out of the
// page so they can be tested on their own (as gallery/order.ts holds the
// Gallery's). Every function here is pure: the page passes in what it reads
// from the hash and its collapsed state.
import { groupNames } from '../grouping'
import { matchWords } from '../matchWords'
import { newestFirst } from '../ui/ListSort.svelte'
import type { AssetFile, AssetLibrary, ShadowedAsset } from '../types'

type Origin = AssetFile['origin']

// What each library is called in the page's own voice. `common` is the
// shared library and `examples` a read-only tree an --examples-dir brought;
// neither name means anything to someone who has not read the server's
// flags
export const LIBRARY_LABELS: Record<Origin, string> = {
  workspace: 'This workspace',
  common: 'Shared library',
  examples: 'Examples',
}

/** Whether `asset` is on this page. A workspace's page shows the shared and
 * examples libraries only as far as its own workflows reach into them - the
 * whole shared library is the Shared assets page's, and listing it in every
 * workspace buried what the workspace made. Those workflows resolve their
 * `asset:` names through the workspace first, so a hidden shared copy is
 * shown when the name is one they use */
export const onPage = (asset: AssetFile, shared: boolean): boolean =>
  shared
    ? asset.origin !== 'workspace'
    : asset.origin === 'workspace' || !!asset.used_by?.length

/** The assets whose name matches every word of `filter` */
export const matchedAssets = (assets: AssetFile[], filter: string) =>
  assets.filter((a) => matchWords(filter, a.name))

/** `matched` in the grid's order: newest first is one list across a
 * library's folders; otherwise the server's order */
export function visibleAssets(
  matched: AssetFile[],
  newest: boolean,
): AssetFile[] {
  if (!newest) return matched
  const byName = new Map(matched.map((a) => [a.name, a]))
  return newestFirst([...byName.keys()], (n) => byName.get(n)?.mtime).map((n) =>
    byName.get(n)!,
  )
}

export interface AssetSection extends AssetLibrary {
  label: string
  assets: AssetFile[]
  shadowed: ShadowedAsset[]
}

/** Whether a section has anything to show */
export const shows = (section: { assets: unknown[]; shadowed: unknown[] }) =>
  section.assets.length > 0 || section.shadowed.length > 0

/** One section per library on the search path, in the order the server
 * resolves them - the path is the page's top level and folders sit inside
 * it, because which library a name is in is what decides whether it can be
 * deleted, what a delete costs, and what it hides. Each holds the
 * `visible` assets on this page from that library, and the entries it
 * hides that match `filter` (on a workspace's page, only the names its
 * workflows use). Unfiltered, an empty library still shows its header, so
 * an empty workspace says where an upload would land - but on a
 * workspace's page a shared library it uses nothing from is left out
 * rather than shown as an empty "used here". While a filter is on, a
 * section with nothing to show is skipped */
export function assetSections(
  libraries: AssetLibrary[],
  visible: AssetFile[],
  shadowed: ShadowedAsset[],
  opts: { shared: boolean; filter: string; usedNames: Set<string> },
): AssetSection[] {
  const { shared, filter, usedNames } = opts
  const filterActive = filter !== ''
  return libraries
    .map((library) => ({
      ...library,
      label:
        (LIBRARY_LABELS[library.origin] ?? library.origin) +
        (shared || library.origin === 'workspace' ? '' : ', used here'),
      assets: visible.filter(
        (a) => a.origin === library.origin && onPage(a, shared),
      ),
      shadowed: shadowed.filter(
        (s) =>
          s.origin === library.origin &&
          matchWords(filter, s.name) &&
          (shared || usedNames.has(s.name)),
      ),
    }))
    .filter((section) => !shared || section.origin !== 'workspace')
    .filter(
      (section) =>
        shows(section) ||
        (!filterActive && (shared || section.origin === 'workspace')),
    )
}

/** What the bulk actions can reach: the grid in the order the page lays it
 * out, minus anything not on screen or not tickable. A shut section is
 * hidden, and a read-only library's tile carries no checkbox and would
 * answer 403 - selecting either is a bulk action touching what the user
 * cannot see or undo from the tile */
export const pickableAssets = (
  sections: AssetSection[],
  isOpen: (origin: string) => boolean,
): AssetFile[] =>
  sections
    .filter((section) => isOpen(section.origin) && section.writable)
    .flatMap((section) => section.assets)

/** The folder heading an asset sits under: none when newest first, which
 * is one list across the folders */
export const folderGroup =
  (folderOf: (name: string) => string | undefined, newest: boolean) =>
  (name: string): string =>
    newest ? '' : (folderOf(name) ?? '')

/** What the viewer steps through in `section`: its assets in the grid's
 * order, minus a shut folder's (`shut`); while filtering, every folder is
 * open, as FolderGroups has it */
export function viewerNames(
  section: AssetSection,
  groupOf: (name: string) => string,
  shut: (folder: string) => boolean,
  filterActive: boolean,
): string[] {
  const grouped = groupNames(
    section.assets.map((a) => a.name),
    groupOf,
  )
  return [...grouped.entries()].flatMap(([g, names]) =>
    !filterActive && shut(g) ? [] : names,
  )
}
