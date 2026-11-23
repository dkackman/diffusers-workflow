// Saving a gallery file to the asset library, under a name later workflows
// can use without depending on the job that made it. The name is asked for
// in the app's own dialog; a name already in the library is never replaced
// without asking, and Keep both saves under freeName's suggestion. Cancel
// or Escape at either question is an answer, not a failure: it sends
// nothing more and reports nothing.
import { ApiError, api } from '../api'
import { chooseDialog, nameDialog } from '../confirm.svelte'
import { freeName } from '../documentSave'
import { MAX_NAME_LENGTH, isNameSegment, leafName } from '../names'
import { ASSET, referenceName } from '../references'
import type { GalleryFile } from '../types'

// How many names Keep both tries before giving up, as uploads.ts does
const KEEP_BOTH_TRIES = 20

/** Why `name` will not do as an asset name, or null */
export function assetNameError(name: string): string | null {
  if (!name) return 'Type a name for the asset'
  if ([...name].length > MAX_NAME_LENGTH)
    return `Use at most ${MAX_NAME_LENGTH} characters`
  if (!isNameSegment(name))
    return 'Start with a letter, digit or _; then letters, digits, _, - and .'
  return null
}

/** keep_output's 409 for a name already in the library. Its 409 also
 * means "no asset library", and the taken one carries no code, so it is
 * told apart by its wording (dw/server/routes/assets.py). */
function nameTaken(e: unknown): boolean {
  if (!(e instanceof ApiError) || e.status !== 409) return false
  return e.code === 'exists' || e.message.includes('already exists')
}

/** The name the server stores: it adds the file's extension to a name
 * typed without one */
function storedName(name: string, file: GalleryFile): string {
  const leaf = leafName(file.name)
  const dot = leaf.lastIndexOf('.')
  if (name.includes('.') || dot <= 0) return name
  return name + leaf.slice(dot)
}

async function kept(
  file: GalleryFile,
  name: string,
  overwrite?: true,
): Promise<{ name: string }> {
  const result = overwrite
    ? await api.keepOutput(file.name, name, true)
    : await api.keepOutput(file.name, name)
  return { name: referenceName(result.reference, ASSET) ?? name }
}

/** Keep both: the first free name, and the next when the listing was
 * behind the library */
async function keepBoth(
  file: GalleryFile,
  name: string,
  existing: string[],
): Promise<{ name: string }> {
  const taken = new Set([...existing, name])
  for (let tries = 1; ; tries++) {
    const free = freeName(name, taken)
    try {
      return await kept(file, free)
    } catch (e) {
      if (!nameTaken(e) || tries >= KEEP_BOTH_TRIES) throw e
      taken.add(free)
    }
  }
}

/** Ask what to do with a name already in the library */
async function onTaken(
  file: GalleryFile,
  name: string,
  existing: string[],
): Promise<{ name: string } | null> {
  const stored = storedName(name, file)
  const free = freeName(stored, new Set([...existing, stored]))
  const question = `"${stored}" is already in the asset library.`
  // Titled in full: the first sentence would end at the name's own dot
  const choice = await chooseDialog(
    question,
    [
      { label: 'Cancel', value: 'cancel' },
      { label: 'Keep both', value: 'keep-both', title: `save it as ${free}` },
      { label: 'Replace', value: 'replace', danger: true },
    ],
    { title: question },
  )
  if (choice === 'replace') return kept(file, name, true)
  if (choice === 'keep-both') return keepBoth(file, stored, existing)
  return null
}

/** Save `file` to the asset library under a name the person types.
 * `existing` is the library's names as last listed, which Keep both's
 * suggestion avoids. Null when they cancel; a failure throws. */
export async function saveAsAsset(
  file: GalleryFile,
  existing: string[],
): Promise<{ name: string } | null> {
  const name = await nameDialog(
    'Later workflows can use the file under this name.',
    {
      title: 'Save as asset',
      label: 'Name in the asset library',
      initial: leafName(file.name),
      confirmLabel: 'Save',
      validate: assetNameError,
    },
  )
  if (name === null) return null
  try {
    return await kept(file, name)
  } catch (e) {
    if (!nameTaken(e)) throw e
    return onTaken(file, name, existing)
  }
}
