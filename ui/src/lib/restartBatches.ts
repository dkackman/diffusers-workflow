import { api } from './api'
import { confirmDialog } from './confirm.svelte'
import { JOBS_PAGE, listJobs } from './jobsListing'
import { describeEstimate, describeEstimateTotal } from './plan'
import { notify } from './toast'
import { workflowName } from './terms'
import type { JobSummary } from './types'

/** One server restart's worth of cancelled jobs still to rerun: the batch
 * key (`restart_batch`, the dead server's instance), one member id the batch
 * routes take, and how many are still to rerun. */
export interface RestartBanner {
  batch: string
  id: string
  count: number
  newest: number
}

type Tombstone = JobSummary & { restart_batch: string }

const isTombstone = (job: JobSummary): job is Tombstone =>
  !!job.restart_batch && job.status === 'cancelled'

const rerunIds = (jobs: JobSummary[]) =>
  new Set(jobs.map((job) => job.rerun_of).filter((id): id is string => !!id))

/** The banners the loaded rows call for, newest batch first, one per batch
 * (#1473). `pending` is the server's count of what each batch has left to
 * rerun, which a page of the listing cannot see whole (#1633). A batch it
 * does not name falls back to the loaded tombstones no loaded row's
 * `rerun_of` names. The batch route's preview stays the authority, and
 * `dismissed` holds the batches a preview found nothing left in. */
export function restartBanners(
  jobs: JobSummary[],
  dismissed: ReadonlySet<string> = new Set(),
  pending: ReadonlyMap<string, number> = new Map(),
): RestartBanner[] {
  const rerun = rerunIds(jobs)
  const banners = new Map<string, RestartBanner>()
  for (const job of jobs.filter(isTombstone)) {
    const batch = job.restart_batch
    const banner = banners.get(batch) ?? {
      batch,
      id: job.id,
      count: 0,
      newest: 0,
    }
    if (!rerun.has(job.id)) banner.count += 1
    banner.newest = Math.max(banner.newest, job.created_at ?? 0)
    banners.set(batch, banner)
  }
  return [...banners.values()]
    .map((banner) => ({
      ...banner,
      count: pending.get(banner.batch) ?? banner.count,
    }))
    .filter((banner) => banner.count > 0 && !dismissed.has(banner.batch))
    .sort((a, b) => b.newest - a.newest)
}

/** The question a Jobs page asks (listJobs), for finding a batch wholly
 * beyond its page. */
export interface BeyondPage {
  ws: string
  scoped: Parameters<typeof listJobs>[1]
}

/** The tombstones among the newest cancelled jobs, where a batch wholly
 * beyond the page is: read once per question, and again after a failure. */
function beyondReader() {
  let older = { question: '', rows: [] as Tombstone[] }
  return async (beyond: BeyondPage | null): Promise<Tombstone[]> => {
    if (!beyond) return []
    const question = JSON.stringify(beyond)
    if (older.question === question) return older.rows
    try {
      const opts = { status: 'cancelled', limit: JOBS_PAGE }
      const { jobs } = await listJobs(beyond.ws, beyond.scoped, false, opts)
      older = { question, rows: jobs.filter(isTombstone) }
      return older.rows
    } catch {
      return []
    }
  }
}

/** What the Jobs page's poll reads for its banners, kept between polls so a
 * batch's size is asked once - again only when the listing shows a rerun it
 * did not, or the banner's own rerun queued one (`forget`). `read` answers
 * the tombstones to offer, loaded or beyond the page, and each batch's
 * `pending`. A failed read is not kept, and the next poll asks again. */
export function restartReader() {
  const sizes = new Map<string, { seen: string; pending: number }>()
  const olderRows = beyondReader()

  async function size(job: Tombstone, seen: string) {
    const kept = sizes.get(job.restart_batch)
    if (kept?.seen === seen) return kept.pending
    try {
      const { pending } = await api.restartBatch(job.id)
      sizes.set(job.restart_batch, { seen, pending })
      return pending
    } catch {
      return undefined
    }
  }

  return {
    forget: (batch: string) => void sizes.delete(batch),
    async read(jobs: JobSummary[], beyond: BeyondPage | null) {
      const loaded = new Set(jobs.map((job) => job.id))
      const unloaded = (await olderRows(beyond)).filter(
        (job) => !loaded.has(job.id),
      )
      const members = new Map(
        [...unloaded, ...jobs]
          .filter(isTombstone)
          .map((job) => [job.restart_batch, job]),
      )
      const seen = [...rerunIds(jobs)].sort().join()
      const pending = new Map<string, number>()
      await Promise.all(
        [...members].map(async ([batch, job]) => {
          const count = await size(job, seen)
          if (count !== undefined) pending.set(batch, count)
        }),
      )
      // a row from beyond the page is offered only once its size is known
      const known = unloaded.filter((job) => pending.has(job.restart_batch!))
      return { tombstones: [...jobs, ...known], pending }
    },
  }
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
    notify.failure('preview the rerun', e)
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
      `• ${entry.workflow || entry.workflow_name ? workflowName(entry) : '(no workflow)'} (${entry.id}, ${entry.workspace}) - ${describeEstimate(entry.estimate).text}`,
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
    notify.failure('rerun the jobs', e)
    return 'cancelled'
  }
  return 'queued'
}
