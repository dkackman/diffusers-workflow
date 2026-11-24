import { expect, it, vi } from 'vitest'
import { ApiError } from './apiErrors'
import {
  firstLine,
  jobMatches,
  listJobs,
  listWithPrompts,
  mergeListings,
  listingRefused,
  listingWorkspace,
  queueLength,
} from './jobsListing'
import type { JobSummary } from './types'

it("a workspace's Jobs page lists that workspace", () => {
  expect(listingWorkspace('workspace', 'other', 'studio', '')).toBe('studio')
  expect(listingWorkspace('workspace', '', 'studio', 'r1')).toBe('studio')
})

it('the all-jobs view lists every workspace, or the one picked', () => {
  expect(listingWorkspace('all', '', 'studio', '')).toBe('')
  expect(listingWorkspace('all', 'shots', 'studio', '')).toBe('shots')
})

it('a request filter never spans workspaces: with none picked it is read in the current one', () => {
  expect(listingWorkspace('all', '', 'studio', 'r1')).toBe('studio')
  expect(listingWorkspace('all', 'shots', 'studio', 'r1')).toBe('shots')
})

it('a refusal empties the list; a fault, an outage or a lock keeps the rows', () => {
  expect(
    listingRefused(new ApiError("Request 'R 1' must match ^[a-z0-9]", 422)),
  ).toBe(true)
  expect(listingRefused(new ApiError('Unknown workspace nope', 404))).toBe(true)
  expect(listingRefused(new ApiError('Internal Server Error', 500))).toBe(false)
  expect(listingRefused(new TypeError('Failed to fetch'))).toBe(false)
  expect(
    listingRefused(new ApiError('Invalid or missing API token', 401)),
  ).toBe(false)
})

const listed = vi.hoisted(() => [] as unknown[][])
const served = vi.hoisted(() => ({
  jobs: [] as unknown[],
  prompts: {} as Record<string, string>,
}))
vi.mock('./api', () => ({
  api: {
    // What the server does with `status`, `limit` and `prompts`
    listJobs: (...args: unknown[]) => {
      listed.push(args)
      const [, limit, status, , , opts] = args as [
        unknown,
        number | undefined,
        string | undefined,
        unknown,
        unknown,
        { prompts?: boolean },
      ]
      const states = status?.split(',')
      const matched = (served.jobs as JobSummary[]).filter(
        (j) => !states || states.includes(j.status),
      )
      return Promise.resolve({
        jobs: limit === undefined ? matched : matched.slice(-limit),
        total: matched.length,
        ...(opts?.prompts ? { prompts: served.prompts } : {}),
      })
    },
  },
}))

const none = { request: '', item: '' }
const job = (id: string, status: string, created_at: number) =>
  ({ id, status, created_at, workflow: 'w' }) as JobSummary
// [status, limit, prompts] of each read since `from`
const reads = (from: number) =>
  listed
    .slice(from)
    .map((a) => [a[2], a[1], (a.at(-1) as { prompts: boolean }).prompts])

it("the Jobs page's listing asks for each job's prompt unless told not to", async () => {
  await listJobs('studio', { request: 'r1', item: 'alpha' })
  expect(listed.at(-1)).toEqual([
    'studio',
    undefined,
    undefined,
    'r1',
    'alpha',
    { prompts: true },
  ])
  await listJobs('studio', { request: '', item: '' }, false)
  expect(listed.at(-1)?.at(-1)).toEqual({ prompts: false })
  await listJobs('', none, true, { status: 'failed', limit: 5 })
  expect(listed.at(-1)).toEqual([
    undefined,
    5,
    'failed',
    undefined,
    undefined,
    { prompts: true },
  ])
})

it('the poll asks for live jobs and the newest finished page', async () => {
  served.jobs = []
  const from = listed.length
  await listWithPrompts('', none, new Map(), 50)
  expect(reads(from)).toEqual(
    expect.arrayContaining([
      ['queued,running', undefined, true],
      ['succeeded,failed,cancelled', 50, true],
    ]),
  )
  expect(listed.length - from).toBe(2)
})

it('a running job older than the page is listed', async () => {
  served.jobs = [
    job('old-run', 'running', 1),
    ...Array.from({ length: 60 }, (_, i) => job(`d${i}`, 'succeeded', 10 + i)),
  ]
  const { list } = await listWithPrompts('', none, new Map(), 50)
  expect(list.jobs).toHaveLength(51)
  expect(list.jobs[0].id).toBe('old-run')
})

it('a job in both answers is listed once, oldest first', () => {
  const merged = mergeListings(
    { jobs: [job('a', 'running', 5)], total: 1 },
    { jobs: [job('b', 'succeeded', 2), job('a', 'succeeded', 5)], total: 2 },
  )
  expect(merged.jobs.map((j) => [j.id, j.status])).toEqual([
    ['b', 'succeeded'],
    ['a', 'running'],
  ])
})

it('total counts the live and every finished job', async () => {
  served.jobs = [
    job('q', 'queued', 100),
    job('r', 'running', 99),
    ...Array.from({ length: 70 }, (_, i) => job(`d${i}`, 'failed', i)),
  ]
  const { list } = await listWithPrompts('', none, new Map(), 50)
  expect(list.total).toBe(72)
  expect(list.finished).toBe(70)
  expect(list.jobs).toHaveLength(52)
})

it('a job that finishes between the two reads is listed, not lost', async () => {
  served.jobs = [job('x', 'running', 1), job('d', 'succeeded', 0)]
  const orig = served.jobs
  // finish x right after the live read is answered
  const finishAfterLive = vi.fn()
  const { api } = await import('./api')
  const real = api.listJobs
  api.listJobs = ((...a: Parameters<typeof real>) => {
    const out = real(...a)
    if (a[2] === 'queued,running')
      served.jobs = orig.map((j) => ({
        ...(j as JobSummary),
        status: 'succeeded',
      }))
    finishAfterLive()
    return out
  }) as typeof real
  try {
    const { list } = await listWithPrompts('', none, new Map(), 50)
    expect(list.jobs.map((j) => j.id)).toContain('x')
    expect(list.total).toBe(2)
  } finally {
    api.listJobs = real
  }
})

it('with held jobs a read is one finished page, merged into what is held', async () => {
  served.jobs = Array.from({ length: 120 }, (_, i) =>
    job(`d${i}`, 'succeeded', i),
  )
  const held = new Map<string, JobSummary>()
  const known = new Map<string, string>()
  const first = await listWithPrompts('', none, known, 120, held)
  expect(first.list.jobs).toHaveLength(120)
  served.jobs = [...served.jobs, job('d120', 'succeeded', 120)]
  const from = listed.length
  const next = await listWithPrompts('', none, known, 120, held)
  // the new job's prompt is asked for again, alone; no read is the history
  expect(reads(from).every((r) => r[1] === undefined || r[1] === 50)).toBe(true)
  expect(next.list.jobs).toHaveLength(121)
  expect(next.list.finished).toBe(121)
})

it('prompts are asked for only for jobs not yet known', async () => {
  served.jobs = [job('j1', 'running', 1), job('j2', 'succeeded', 2)]
  served.prompts = { j1: 'a fox', j2: 'a whale' }
  const known = new Map<string, string>()
  const first = await listWithPrompts('', none, known, 50)
  expect(first.prompts).toEqual({ j1: 'a fox', j2: 'a whale' })
  const from = listed.length
  await listWithPrompts('', none, known, 50)
  expect(reads(from).every(([, , prompts]) => prompts === false)).toBe(true)
  expect(listed.length - from).toBe(2)
})

it('a new job on the second poll re-asks for prompts', async () => {
  served.jobs = [job('j1', 'running', 1), job('j2', 'succeeded', 2)]
  served.prompts = { j1: 'a fox', j2: 'a whale', j3: 'an owl' }
  const known = new Map<string, string>()
  await listWithPrompts('', none, known, 50)
  served.jobs = [...served.jobs, job('j3', 'queued', 3)]
  const from = listed.length
  const { prompts } = await listWithPrompts('', none, known, 50)
  // each read on its own: the live read saw the new job, the finished one did not
  expect(reads(from)).toEqual(
    expect.arrayContaining([
      ['queued,running', undefined, false],
      ['queued,running', undefined, true],
      ['succeeded,failed,cancelled', 50, false],
    ]),
  )
  expect(listed.length - from).toBe(3)
  expect(prompts.j3).toBe('an owl')
  expect(prompts.j1).toBe('a fox')
})

it('a search matches word by word across name, id, note and prompt', () => {
  const job = {
    id: 'job-77',
    workflow: 'wf/text-to-image.json',
    note: 'Keeper',
  } as JobSummary
  expect(jobMatches(job, 'a red fox', 'fox 77')).toBe(true)
  expect(jobMatches(job, 'a red fox', 'keeper image')).toBe(true)
  expect(jobMatches(job, 'a red fox', 'whale')).toBe(false)
  expect(jobMatches(job, '', '  ')).toBe(true)
})

it('the first line of a prompt skips blank lines', () => {
  expect(firstLine('\n  a fox \nmore')).toBe('a fox')
  expect(firstLine('')).toBe('')
})

const at = (...positions: Array<number | undefined>) =>
  positions.map((queue_position) => ({ queue_position })) as JobSummary[]

it('the queue is as long as the server says, when it says', () => {
  // One request's job, third of five in the whole queue
  expect(queueLength(at(2, undefined), 5)).toBe(5)
})

it('without a figure it is at least one past the highest position seen', () => {
  expect(queueLength(at(0, 3, undefined))).toBe(4)
  expect(queueLength(at(undefined))).toBe(0)
  expect(queueLength([])).toBe(0)
})

it('a figure the rows have already outrun does not shorten the queue', () => {
  expect(queueLength(at(4), 3)).toBe(5)
})
