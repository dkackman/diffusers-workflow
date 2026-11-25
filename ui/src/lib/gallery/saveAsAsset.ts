// Saving a gallery file to the asset library, under a name later workflows
// can use without depending on the job that made it. The name is asked for
// in the app's own dialog; a name already in the library is never replaced
// without asking, and Keep both saves under freeName's suggestion. The
// library is listed only when the name is taken, for that suggestion, and a
// failed listing is no names: the editors' shared list (assetlib) is never
// touched (#1600). Cancel
// or Escape at either question is an answer, not a failure: it sends
// nothing more and reports nothing.
import { ApiError, api } from '../api'
import { chooseDialog, nameDialog } from '../confirm.svelte'
import { freeName } from '../documentSave'
import { MAX_NAME_LENGTH, isNameSegment, leafName } from '../names'
import { ASSET, referenceName } from '../references'
/** What a save reads of the file: only its output name */
type Saved = { name: string }

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
 * means "no asset library"; the taken one carries `code: 'exists'`
 * (dw/server/routes/assets.py). */
function nameTaken(e: unknown): boolean {
  if (!(e instanceof ApiError) || e.status !== 409) return false
  return e.code === 'exists'
}

/** The name the server stores: it adds the file's extension to a name
 * typed without one */
function storedName(name: string, file: Saved): string {
  const leaf = leafName(file.name)
  const dot = leaf.lastIndexOf('.')
  if (name.includes('.') || dot <= 0) return name
  return name + leaf.slice(dot)
}

async function kept(
  file: Saved,
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
  file: Saved,
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

/** The library's names, read for Keep both's suggestion */
export const listAssetNames = (): Promise<string[]> =>
  api.listAssets().then((result) => result.assets.map((asset) => asset.name))

/** Ask what to do with a name already in the library */
async function onTaken(
  file: Saved,
  name: string,
  listNames: () => Promise<string[]>,
): Promise<{ name: string } | null> {
  // A listing that fails leaves Keep both to find the free name by trying
  const existing = await listNames().catch((): string[] => [])
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
 * `listNames` reads the library's names, which Keep both's suggestion
 * avoids, once the name turns out taken. Null when they cancel; a failure
 * throws. */
export async function saveAsAsset(
  file: Saved,
  listNames: () => Promise<string[]> = listAssetNames,
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
    return onTaken(file, name, listNames)
  }
}

/** The Save as asset button's copy, the same in the Gallery and the job
 * viewer: `action(saveAsAsset, SAVING_ASSET)` */
export const SAVING_ASSET = {
  doing: 'save it as an asset',
  success: (r: { name: string }) => `Saved as asset ${r.name}`,
}
