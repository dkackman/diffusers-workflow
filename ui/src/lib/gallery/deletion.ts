// The wording and the partition behind deleting a ticked selection: starred
// files are the ones a person marked to keep, so the delete says how many
// are among them and can spare them.
import type { ConfirmSpec } from '../action.svelte'
import { chooseDialog, confirmDialog } from '../confirm.svelte'
import { destroyConfirm } from '../destructive'
import type { GalleryFile } from '../types'

export function deletionChoices(
  files: GalleryFile[],
  names: string[],
): ConfirmSpec & { starred: string[] } {
  const picked = new Set(names)
  const starred = files
    .filter((f) => picked.has(f.name) && f.starred)
    .map((f) => f.name)
  const spec = destroyConfirm({
    count: names.length,
    noun: 'file',
    starred: starred.length,
  })
  return { ...spec, starred }
}

/** The confirm for deleting the one open file, starred or not */
export const fileConfirm = (file: GalleryFile): ConfirmSpec =>
  destroyConfirm({
    name: file.name,
    noun: 'file',
    starred: file.starred ? 1 : 0,
  })

/** Asks what to delete from a ticked selection: the names to remove, or
 * null when the person backed out. */
export async function askDeletion(
  files: GalleryFile[],
  names: string[],
): Promise<string[] | null> {
  const { starred, ...spec } = deletionChoices(files, names)
  if (starred.length === 0)
    return (await confirmDialog(spec.message, spec)) ? names : null
  const unstarred = names.length - starred.length
  // Every ticked file starred: there is nothing to spare them from
  const answer = await chooseDialog(
    spec.message,
    [
      { label: 'Delete all', value: 'all', danger: true },
      ...(unstarred > 0
        ? [{ label: `Delete the ${unstarred} unstarred`, value: 'unstarred' }]
        : []),
      { label: 'Cancel', value: 'cancel' },
    ],
    { title: spec.title },
  )
  if (answer === 'all') return names
  if (answer !== 'unstarred') return null
  const spare = new Set(starred)
  const rest = names.filter((n) => !spare.has(n))
  return rest.length ? rest : null
}
