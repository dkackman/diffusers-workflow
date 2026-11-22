// The wording and the partition behind deleting a ticked selection: starred
// files are the ones a person marked to keep, so the delete says how many
// are among them and can spare them.
import { chooseDialog, confirmDialog } from '../confirm.svelte'
import type { GalleryFile } from '../types'

export function deletionChoices(
  files: GalleryFile[],
  names: string[],
): { message: string; starred: string[] } {
  const picked = new Set(names)
  const starred = files
    .filter((f) => picked.has(f.name) && f.starred)
    .map((f) => f.name)
  const n = names.length
  const noun = `${n} file${n === 1 ? '' : 's'}`
  if (starred.length === 0)
    return { message: `Delete ${noun}? This removes them on disk.`, starred }
  const s = starred.length
  return {
    message: `Delete ${noun}? ${s} of them ${s === 1 ? 'is' : 'are'} starred. This removes them on disk.`,
    starred,
  }
}

/** Asks what to delete from a ticked selection: the names to remove, or
 * null when the person backed out. */
export async function askDeletion(
  files: GalleryFile[],
  names: string[],
): Promise<string[] | null> {
  const { message, starred } = deletionChoices(files, names)
  if (starred.length === 0) {
    const ok = await confirmDialog(message, {
      confirmLabel: 'Delete',
      danger: true,
    })
    return ok ? names : null
  }
  const unstarred = names.length - starred.length
  // Every ticked file starred: there is nothing to spare them from
  const answer = await chooseDialog(message, [
    { label: 'Delete all', value: 'all', danger: true },
    ...(unstarred > 0
      ? [{ label: `Delete the ${unstarred} unstarred`, value: 'unstarred' }]
      : []),
    { label: 'Cancel', value: 'cancel' },
  ])
  if (answer === 'all') return names
  if (answer !== 'unstarred') return null
  const spare = new Set(starred)
  const rest = names.filter((n) => !spare.has(n))
  return rest.length ? rest : null
}
