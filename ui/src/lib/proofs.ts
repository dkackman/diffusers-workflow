import type { GalleryFile } from './types'

/** The newest output each workflow has produced, by workflow identity. A run
 * writes to <outputs>/<identity>/<run id>/, and the gallery listing reports
 * that identity as an entry's `folder` with the run id already stripped, so
 * a workflow's name matches its folder directly. Entries arrive newest
 * first, so the first one seen for a folder is that workflow's latest.
 * Images win over video, which shows as its first frame - an image is
 * what the workflow made, a frame is a sample of it. Audio and text have nothing to show in a frame, so a workflow with
 * neither an image nor a video gets no proof. */
export function latestProofs(
  files: GalleryFile[],
): Record<string, GalleryFile> {
  return proofsBy(files, (file) => file.folder)
}

/** Each run's proof, by run id: its first image, else its first clip, by
 * the rule `latestProofs` keeps per workflow - what a job's row shows. */
export function proofByRun(files: GalleryFile[]): Record<string, GalleryFile> {
  return proofsBy(files, (file) => file.run_id)
}

function proofsBy(
  files: GalleryFile[],
  keyOf: (file: GalleryFile) => string | null | undefined,
): Record<string, GalleryFile> {
  const proofs: Record<string, GalleryFile> = {}
  for (const file of files) {
    const key = keyOf(file)
    if (!key) continue
    if (file.kind !== 'image' && file.kind !== 'video') continue
    const held = proofs[key]
    if (!held || (held.kind !== 'image' && file.kind === 'image'))
      proofs[key] = file
  }
  return proofs
}
