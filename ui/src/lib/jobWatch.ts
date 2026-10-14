/** What the tab title and the finish alert read off the 5-second health
 * poll. A job runs for minutes to half an hour, usually in a tab nobody is
 * looking at, so the title is where its state has to be. */
import { formatDuration } from './format'
import type { HealthInfo, JobDetail } from './types'

export const BASE_TITLE = 'diffusers-workflow'

export type TitleJob = Pick<
  JobDetail,
  'workflow' | 'workflow_name' | 'progress'
>

/** Every job a card is running now; `current_job` alone on a server too old
 * to report `workers`. */
export function runningIds(health: HealthInfo | null): string[] {
  if (!health) return []
  const ids = (health.workers ?? [])
    .map((w) => w.current_job)
    .filter((id): id is string => !!id)
  if (health.current_job && !ids.includes(health.current_job))
    ids.push(health.current_job)
  return ids
}

export function endedSince(
  before: readonly string[],
  now: readonly string[],
): string[] {
  return before.filter((id) => !now.includes(id))
}

/** 0-99: steps behind it plus how far the current step's denoise loop is.
 * Never 100 - a running job is not done. Null with nothing to go on. */
export function percentDone(progress: JobDetail['progress']): number | null {
  if (!progress?.total_steps) return null
  const inner = progress.denoise_total_steps
    ? Math.min(1, (progress.denoise_step ?? 0) / progress.denoise_total_steps)
    : 0
  const done = ((progress.step_index ?? 0) + inner) / progress.total_steps
  return Math.min(99, Math.max(0, Math.floor(done * 100)))
}

const leaf = (job: TitleJob) =>
  (job.workflow ?? job.workflow_name ?? 'job').split('/').pop() || 'job'

export function tabTitle(job: TitleJob | null, othersRunning = 0): string {
  if (!job) return BASE_TITLE
  const pct = percentDone(job.progress)
  const more = othersRunning > 0 ? ` +${othersRunning}` : ''
  return `▶ ${pct === null ? '' : `${pct}% · `}${leaf(job)}${more} — ${BASE_TITLE}`
}

const MARKS: Record<string, string> = {
  succeeded: '✓',
  failed: '✗',
  cancelled: '⊘',
}

export function endedTitle(job: TitleJob & { status: string }): string {
  return `${MARKS[job.status] ?? '•'} ${leaf(job)} — ${BASE_TITLE}`
}

/** "ref-to-video succeeded in 12m 54s" - the toast and the alert body. */
export function finishedText(job: JobDetail): string {
  const took =
    job.started_at && job.finished_at
      ? ` in ${formatDuration(job.finished_at - job.started_at)}`
      : ''
  return `${leaf(job)} ${job.status}${took}`
}
