import { api } from './api'
import { confirmDialog } from './confirm.svelte'
import { describeEstimate, describeEstimateTotal } from './plan'
import { notify } from './toast'
import type { JobSummary } from './types'

/** One server restart's worth of cancelled jobs that look not yet rerun:
 * the batch key (`restart_batch`, the dead server's instance), one member
 * id the batch route takes, and how many members have no rerun in view. */
export interface RestartBanner {
  batch: string
  id: string
  count: number
  newest: number
}

/** The banners the loaded rows call for, newest batch first, one per batch
 * (#1473). A tombstone counts unless some loaded row's `rerun_of` names it.
 * This is an estimate - the list is capped and a rerun may be in a
 * workspace since deleted - so the batch route's preview is the authority,
 * and `dismissed` holds the batches a preview found nothing left in. */
export function restartBanners(
  jobs: JobSummary[],
  dismissed: ReadonlySet<string> = new Set(),
): RestartBanner[] {
  const rerun = new Set(
    jobs.map((job) => job.rerun_of).filter((id): id is string => !!id),
  )
  const banners = new Map<string, RestartBanner>()
  for (const job of jobs) {
    const batch = job.restart_batch
    if (!batch || job.status !== 'cancelled' || rerun.has(job.id)) continue
    if (dismissed.has(batch)) continue
    const created = job.created_at ?? 0
    const banner = banners.get(batch)
    if (banner) {
      banner.count += 1
      banner.newest = Math.max(banner.newest, created)
    } else {
      banners.set(batch, { batch, id: job.id, count: 1, newest: created })
    }
  }
  return [...banners.values()].sort((a, b) => b.newest - a.newest)
}

const plural = (n: number, one: string, many: string) =>
  `${n} ${n === 1 ? one : many}`

/** Preview the batch `id` belongs to, show what a rerun would queue with
 * each estimate and a total, and queue it only on confirm. Answers 'empty'
 * when the preview found nothing left to rerun - the caller hides that
 * batch's banner - 'queued' after a confirmed rerun, and 'cancelled' when
 * nothing was queued (a cancel, or a failed call, already reported). */
export async function rerunRestartBatchWithConfirm(
  id: string,
): Promise<'queued' | 'cancelled' | 'empty'> {
  let preview
  try {
    preview = await api.rerunRestartBatch(id)
  } catch (e) {
    notify.error(e instanceof Error ? e.message : String(e))
    return 'cancelled'
  }
  const would = preview.would_queue ?? []
  const skipped = preview.skipped.map(
    (entry) => `• ${entry.id}: ${entry.reason}`,
  )
  const skippedBlock = skipped.length
    ? `\n\nSkipped:\n${skipped.join('\n')}`
    : ''
  if (!would.length) {
    await confirmDialog(
      `Nothing is left to rerun from this server restart.${skippedBlock}`,
      { confirmLabel: 'Close', alert: true },
    )
    return 'empty'
  }
  const lines = would.map(
    (entry) =>
      `• ${entry.workflow ?? '(no workflow)'} (${entry.id}, ${entry.workspace}) - ${describeEstimate(entry.estimate).text}`,
  )
  const total = describeEstimateTotal(would.map((entry) => entry.estimate))
  const message =
    `Rerun ${plural(would.length, 'job', 'jobs')} a server restart cancelled?` +
    `\n\nWould queue:\n${lines.join('\n')}\n${total.text}${skippedBlock}`
  if (
    !(await confirmDialog(message, { confirmLabel: `Rerun ${would.length}` }))
  )
    return 'cancelled'
  try {
    const answer = await api.rerunRestartBatch(id, true)
    const skippedNote = answer.skipped.length
      ? `; skipped ${answer.skipped.length}: ${answer.skipped
          .map((entry) => `${entry.id} (${entry.reason})`)
          .join(', ')}`
      : ''
    notify.success(`Queued ${answer.queued.length}${skippedNote}`)
  } catch (e) {
    notify.error(e instanceof Error ? e.message : String(e))
    return 'cancelled'
  }
  return 'queued'
}
