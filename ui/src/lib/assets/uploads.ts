// Uploading files to the asset library. Each file keeps its own name, an
// existing asset is never replaced without asking (one dialog per batch),
// and a file the server would refuse is never sent.

import { api } from '../api'
import { UPLOAD_EXTENSIONS } from '../assetlib.svelte'
import { chooseDialog } from '../confirm.svelte'
import { freeName } from '../documentSave'
import { describeError } from '../apiErrors'
import { notify } from '../toast'
import type { Uploaded } from '../types'

export { UPLOAD_EXTENSIONS }

export type ConflictChoice = 'replace' | 'keep-both' | 'skip'

export type UploadResult = {
  uploaded: Uploaded[]
  skipped: string[]
  failed: { name: string; error: unknown }[]
}

// How many names Keep both tries for one file before giving up on it
const KEEP_BOTH_TRIES = 20

export function isUploadable(file: File): boolean {
  const dot = file.name.lastIndexOf('.')
  return (
    dot >= 0 && UPLOAD_EXTENSIONS.includes(file.name.slice(dot).toLowerCase())
  )
}

function isExists(error: unknown): boolean {
  return (
    typeof error === 'object' &&
    error !== null &&
    (error as { status?: unknown }).status === 409 &&
    (error as { code?: unknown }).code === 'exists'
  )
}

/** The listing read once more after the page's own write, for the page to
 * hold as read (resource's `adopt`) rather than reload loudly */
export const readListingAfterWrite = () => api.listAssets()

type Entry = { file: File; name: string }

export async function uploadAll(
  files: File[],
  opts: { shared: boolean; folder?: string; taken?: ReadonlySet<string> },
): Promise<UploadResult> {
  const result: UploadResult = { uploaded: [], skipped: [], failed: [] }
  const unsupported = files.filter((f) => !isUploadable(f))
  if (unsupported.length)
    notify.error(
      `Skipped ${unsupported.map((f) => f.name).join(', ')}: not an image, ` +
        `video, audio or LUT file`,
    )
  const prefix = opts.folder ? `${opts.folder.replace(/\/+$/, '')}/` : ''
  const sendable = files.filter(isUploadable)
  // The listing names a library's assets as uploads/<name>; the names sent
  // and compared here are bare
  const taken = new Set(
    [...(opts.taken ?? [])].map((n) => n.replace(/^uploads\//, '')),
  )
  const existing: Entry[] = []

  for (const file of sendable) {
    const name = prefix + file.name
    try {
      result.uploaded.push(
        await api.uploadMedia(file, name, opts.shared, { createOnly: true }),
      )
    } catch (error) {
      if (isExists(error)) existing.push({ file, name })
      else result.failed.push({ name: file.name, error })
    }
  }
  if (existing.length)
    await resolveExisting(existing, sendable.length, opts.shared, taken, result)
  return result
}

async function resolveExisting(
  existing: Entry[],
  total: number,
  shared: boolean,
  taken: Set<string>,
  result: UploadResult,
): Promise<void> {
  const choice = await chooseDialog<ConflictChoice>(
    `${existing.length} of ${total} files already exist in the library.`,
    [
      { label: 'Replace them', value: 'replace', danger: true },
      { label: 'Keep both', value: 'keep-both' },
      { label: 'Skip them', value: 'skip' },
    ],
  )
  for (const entry of existing) {
    if (choice === 'replace') await attempt(result, entry, entry.name, shared)
    else if (choice === 'keep-both')
      await keepBoth(result, entry, shared, taken)
    else result.skipped.push(entry.file.name)
  }
}

async function attempt(
  result: UploadResult,
  { file }: Entry,
  name: string,
  shared: boolean,
): Promise<void> {
  try {
    result.uploaded.push(await api.uploadMedia(file, name, shared, undefined))
  } catch (error) {
    result.failed.push({ name: file.name, error })
  }
}

async function keepBoth(
  result: UploadResult,
  entry: Entry,
  shared: boolean,
  taken: Set<string>,
): Promise<void> {
  taken.add(entry.name)
  for (let tries = 0; tries < KEEP_BOTH_TRIES; tries++) {
    const name = freeName(entry.name, taken)
    try {
      result.uploaded.push(
        await api.uploadMedia(entry.file, name, shared, { createOnly: true }),
      )
      taken.add(name)
      return
    } catch (error) {
      if (!isExists(error)) {
        result.failed.push({ name: entry.file.name, error })
        return
      }
      taken.add(name)
    }
  }
  result.failed.push({
    name: entry.file.name,
    error: new Error('no free name found'),
  })
}

/** The toasts for a finished batch: what went up (a lone file by its
 * reference), what was left alone, and what failed, as "name: reason". */
export function reportUploads(result: UploadResult): void {
  const { uploaded, skipped, failed } = result
  if (uploaded.length === 1)
    notify.success(`Uploaded as ${uploaded[0].reference ?? uploaded[0].url}`)
  else if (uploaded.length > 1) {
    const total = uploaded.length + skipped.length + failed.length
    notify.success(`Uploaded ${uploaded.length} of ${total} files`)
  }
  if (skipped.length)
    notify.success(
      `Skipped ${skipped.length} existing ${skipped.length === 1 ? 'file' : 'files'}: ${skipped.join(', ')}`,
    )
  if (failed.length)
    notify.error(
      failed
        .map((f) => `${f.name}: ${describeError(f.error).title}`)
        .join('\n'),
    )
}
