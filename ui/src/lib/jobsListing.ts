// What the Jobs page asks the server for, and how it reads the answer: the
// workspace a listing is scoped to, what a failed listing says, and how
// long the waiting queue is when the page holds only some of its rows.
import { api } from './api'
import { ApiError } from './apiErrors'
import type { RequestFilter } from './requestFilter'
import { workflowName } from './terms'
import type { JobList, JobSummary } from './types'

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

/** How many finished jobs the Jobs page reads at first, and each
 * Show older adds. */
export const JOBS_PAGE = 50

/** The listing a Jobs page's filters name: `ws` ('' for every workspace)
 * narrowed to a request, or one item of it, with each job's submitted
 * prompt (`prompts`) when asked, and to the `status` set / newest `limit`
 * when given. */
export const listJobs = (
  ws: string,
  scoped: RequestFilter,
  prompts = true,
  opts: { status?: string; limit?: number } = {},
) =>
  api.listJobs(
    ws || undefined,
    opts.limit,
    opts.status,
    scoped.request || undefined,
    scoped.item || undefined,
    { prompts },
  )

/** The live jobs and the finished page as one listing: by creation, oldest
 * first, each id once (the live answer wins), `total` every live job and
 * every finished one. */
export function mergeListings(live: JobList, done: JobList) {
  const byId = new Map<string, JobSummary>()
  for (const job of [...done.jobs, ...live.jobs]) byId.set(job.id, job)
  const jobs = [...byId.values()].sort(
    (a, b) => (a.created_at ?? 0) - (b.created_at ?? 0),
  )
  const prompts =
    live.prompts || done.prompts
      ? { ...done.prompts, ...live.prompts }
      : undefined
  return {
    jobs,
    total: live.jobs.length + done.total,
    ...(prompts ? { prompts } : {}),
  }
}

/** One read of the listing, asking for prompts when it is the first or
 * names a job the page does not know yet - then it is asked again, alone. */
async function readWithPrompts(
  ws: string,
  scoped: RequestFilter,
  known: Map<string, string>,
  opts: { status: string; limit?: number },
) {
  const first = known.size === 0
  let list = await listJobs(ws, scoped, first, opts)
  if (!first && list.jobs.some((job) => !known.has(job.id)))
    list = await listJobs(ws, scoped, true, opts)
  return list
}

/** The page of the listing - every queued and running job and the newest
 * `limit` finished ones - and each job's prompt, without paying for
 * prompts on every poll: a job's prompt never changes, so `known` (job id
 * -> prompt, kept by the page) holds what was read, and a read is asked
 * again with `prompts` only when a job it names is not in it. */
export async function listWithPrompts(
  ws: string,
  scoped: RequestFilter,
  known: Map<string, string>,
  limit: number = JOBS_PAGE,
) {
  const [live, done] = await Promise.all([
    readWithPrompts(ws, scoped, known, { status: 'queued,running' }),
    readWithPrompts(ws, scoped, known, {
      status: 'succeeded,failed,cancelled',
      limit,
    }),
  ])
  const list = mergeListings(live, done)
  for (const job of list.jobs)
    if (list.prompts || !known.has(job.id))
      known.set(job.id, list.prompts?.[job.id] ?? known.get(job.id) ?? '')
  return { list, prompts: Object.fromEntries(known) as Record<string, string> }
}

/** The first non-empty line of a prompt or note, trimmed */
export const firstLine = (text: string) =>
  text
    .split('\n')
    .find((line) => line.trim())
    ?.trim() ?? ''

/** Whether every word of `query` is in the job's name, id, note or prompt */
export function jobMatches(
  job: JobSummary,
  prompt: string,
  query: string,
): boolean {
  const hay = [workflowName(job), job.id, job.note ?? '', prompt]
    .join('\n')
    .toLowerCase()
  return query
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((word) => hay.includes(word))
}

/** Whole percent of each running job's denoise loop, from its own detail:
 * a job with no loop yet, or one that cannot be read, has none. */
export async function runningProgress(
  rows: JobSummary[],
): Promise<Record<string, number>> {
  const out: Record<string, number> = {}
  await Promise.all(
    rows
      .filter((job) => job.status === 'running')
      .map(async (job) => {
        const p = (await api.getJob(job.id).catch(() => null))?.progress
        if (p?.denoise_step != null && p.denoise_total_steps)
          out[job.id] = Math.min(
            100,
            Math.round((100 * p.denoise_step) / p.denoise_total_steps),
          )
      }),
  )
  return out
}

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
