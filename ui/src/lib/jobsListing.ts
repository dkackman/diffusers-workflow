// What the Jobs page asks the server for, and how it reads the answer: the
// workspace a listing is scoped to, what a failed listing says, and how
// long the waiting queue is when the page holds only some of its rows.
import { api } from './api'
import { ApiError } from './apiErrors'
import type { RequestFilter } from './requestFilter'
import type { JobSummary } from './types'

/** The workspace a Jobs listing asks for; '' is every workspace. A
 * workspace's own page lists `current`. The all-jobs view lists the one
 * picked (`picked`), or all of them - except under a request filter: a
 * request id names a request inside one workspace, and the same id in
 * another is another request, so with none picked it is read in `current`. */
export function listingWorkspace(
  scope: 'workspace' | 'all',
  picked: string,
  current: string,
  request: string,
): string {
  if (scope === 'workspace') return current
  return picked || (request ? current : '')
}

/** Whether a failed listing is the server turning the question down (a
 * `req` that is not a request id, a workspace that does not exist) rather
 * than a fault or an unreachable server. A refusal's message is the
 * explanation, and the rows of an earlier question must not stay under this
 * one's filters; a fault keeps the rows it had, and so does a lock (401),
 * which is the connection's state, not an answer to the question. */
export function listingRefused(error: unknown): boolean {
  return error instanceof ApiError && error.status < 500 && error.status !== 401
}

/** How many jobs are waiting. `queue_position` is a job's place in the
 * whole queue, so a page narrowed to a workspace or a request cannot count
 * its own rows: `queued` is the server's count of every queued job, and
 * without it the queue is at least one past the highest position loaded. */
export function queueLength(jobs: JobSummary[], queued = 0): number {
  const seen = jobs.reduce(
    (highest, job) => Math.max(highest, (job.queue_position ?? -1) + 1),
    0,
  )
  return Math.max(seen, queued)
}

/** The listing a Jobs page's filters name: `ws` ('' for every workspace)
 * narrowed to a request, or one item of it */
export const listJobs = (ws: string, scoped: RequestFilter) =>
  api.listJobs(
    ws || undefined,
    undefined,
    undefined,
    scoped.request || undefined,
    scoped.item || undefined,
  )

/** The server's count of every queued job, asked for only by a narrowed
 * listing that shows a waiting one: an unnarrowed listing holds the whole
 * queue, and 0 leaves the length to the rows. */
export async function queuedEverywhere(
  rows: JobSummary[],
  narrowed: boolean,
): Promise<number> {
  if (!narrowed || !rows.some((job) => job.queue_position !== undefined))
    return 0
  return (await api.listJobs(undefined, 1, 'queued')).total
}
