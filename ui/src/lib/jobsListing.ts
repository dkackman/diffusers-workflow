// What the Jobs page asks the server for, and how it reads the answer: the
// workspace a listing is scoped to, what a failed listing says, and how
// long the waiting queue is when the page holds only some of its rows.
import { ApiError } from './apiErrors'
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

/** What the page says when a listing fails. `refused` is the server
 * turning the question down (a `req` that is not a request id, a workspace
 * that does not exist): its message is the explanation, and the rows of an
 * earlier question must not stay under this one's filters. */
export function listingFailure(error: unknown): {
  message: string
  refused: boolean
} {
  const text = error instanceof Error ? error.message : String(error)
  if (!(error instanceof ApiError))
    return { message: `Could not reach the server: ${text}`, refused: false }
  if (error.status >= 500)
    return {
      message: `The server could not list the jobs: ${text}`,
      refused: false,
    }
  return { message: text, refused: true }
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
