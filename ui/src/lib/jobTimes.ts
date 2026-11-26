import type { JobSummary } from './types'

/** Seconds from `start` to `end`, never negative: a clock a little behind
 * the server's must not read '-4.0s' */
export const elapsed = (start: number, end: number): number =>
  Math.max(0, end - start)

/** Seconds a job waited for a card: submit to start, or submit to now for
 * one still waiting. Null when there is no wait to speak of - no clock, or
 * cancelled before it ever started. */
export function queueWait(
  job: Pick<JobSummary, 'created_at' | 'started_at' | 'status'>,
  now: number,
): number | null {
  if (!job.created_at) return null
  if (job.started_at) return elapsed(job.created_at, job.started_at)
  return job.status === 'queued' ? elapsed(job.created_at, now) : null
}
