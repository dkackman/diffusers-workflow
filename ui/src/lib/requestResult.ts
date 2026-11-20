import { groupResultFiles } from './results'
import type { JobDetail, LedgerItemStatus } from './types'

const IMAGE_EXT = /\.(png|jpe?g|webp|gif)$/i
const VIDEO_EXT = /\.(mp4|webm|mov|mkv)$/i

/** A file that can stand for a job as a picture: an image as itself, a
 * video as its first frame. */
export type ResultProof = { file: string; kind: 'image' | 'video' }

/** The one file that stands for a job in a request, or null when the job
 * made nothing to look at (audio, text, no output). The final steps' files
 * come before any other step's - a keyframe an earlier step drew is not
 * what a clip's run was for - and within either an image before a video,
 * as the workflow proofs choose (proofs.ts). A run whose manifest names no
 * roles is read as one tier. */
export function resultProof(job: JobDetail): ResultProof | null {
  const kindOf = (file: string) =>
    job.output_kinds?.[file] ??
    (IMAGE_EXT.test(file) ? 'image' : VIDEO_EXT.test(file) ? 'video' : null)
  const groups = groupResultFiles(job.manifest, [])
  const tiers = [
    groups.filter((g) => g.role === 'final'),
    groups.filter((g) => g.role !== 'final'),
  ]
  for (const tier of tiers) {
    const files = tier.flatMap((g) => g.files)
    for (const kind of ['image', 'video'] as const) {
      const file = files.find((f) => kindOf(f) === kind)
      if (file) return { file, kind }
    }
  }
  return null
}

/** "2 done · 1 planned": the counts in the order a plan reads. */
const STATUS_ORDER: LedgerItemStatus[] = [
  'done',
  'running',
  'queued',
  'planned',
  'failed',
  'dropped',
]
export function statusCounts(counts: Record<string, number>): string {
  const known = STATUS_ORDER.filter((s) => counts[s])
  const rest = Object.keys(counts).filter(
    (s) => counts[s] && !(STATUS_ORDER as string[]).includes(s),
  )
  return [...known, ...rest].map((s) => `${counts[s]} ${s}`).join(' · ')
}
