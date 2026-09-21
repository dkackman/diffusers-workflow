import type { GalleryFile } from './types'

/** The newest output each workflow has produced, by workflow identity. A run
 * writes to <outputs>/<identity>/<run id>/, and the gallery listing reports
 * that identity as an entry's `folder` with the run id already stripped, so
 * a workflow's name matches its folder directly. Entries arrive newest
 * first, so the first one seen for a folder is that workflow's latest.
 * Images win over video because only images have a thumbnail endpoint; a
 * video-only workflow falls back to its video, which renders its first
 * frame. */
export function latestProofs(
  files: GalleryFile[],
): Record<string, GalleryFile> {
  const latest: Record<string, GalleryFile> = {}
  for (const file of files) {
    if (!file.folder) continue
    const held = latest[file.folder]
    if (!held) latest[file.folder] = file
    else if (held.kind !== 'image' && file.kind === 'image')
      latest[file.folder] = file
  }
  return latest
}
