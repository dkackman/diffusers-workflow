import { groupResultFiles } from './results'
import type { JobDetail, LedgerItemStatus } from './types'

const IMAGE_EXT = /\.(png|jpe?g|webp|gif)$/i

/** The one image that stands for a job in a request: the first image of its
 * final step (or of any step, for a run whose manifest names no roles). Null
 * when the job made none. */
export function firstImage(job: JobDetail): string | null {
  const groups = groupResultFiles(job.manifest, [])
  const ordered = [
    ...groups.filter((g) => g.role === 'final'),
    ...groups.filter((g) => g.role !== 'final'),
  ]
  for (const group of ordered) {
    const found = group.files.find(
      (file) =>
        (job.output_kinds?.[file] ??
          (IMAGE_EXT.test(file) ? 'image' : null)) === 'image',
    )
    if (found) return found
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
