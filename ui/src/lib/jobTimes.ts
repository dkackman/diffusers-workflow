import type { JobSummary } from './types'

/** Seconds a job waited for a card: submit to start, or submit to now for
 * one still waiting. Null when there is no wait to speak of - no clock, or
 * cancelled before it ever started. */
export function queueWait(
  job: Pick<JobSummary, 'created_at' | 'started_at' | 'status'>,
  now: number,
): number | null {
  if (!job.created_at) return null
  if (job.started_at) return Math.max(0, job.started_at - job.created_at)
  return job.status === 'queued' ? Math.max(0, now - job.created_at) : null
}
