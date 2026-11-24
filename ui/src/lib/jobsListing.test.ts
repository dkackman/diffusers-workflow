import { expect, it, vi } from 'vitest'
import { ApiError } from './apiErrors'
import {
  firstLine,
  jobMatches,
  listJobs,
  listWithPrompts,
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
vi.mock('./api', () => ({
  api: {
    listJobs: (...args: unknown[]) => {
      listed.push(args)
      return Promise.resolve({ jobs: [], total: 0 })
    },
  },
}))

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
})

it('a poll asks for prompts only when a job is new to the page', async () => {
  const known = new Map<string, string>()
  const none = { request: '', item: '' }
  const first = await listWithPrompts('', none, known)
  expect(listed.at(-1)?.at(-1)).toEqual({ prompts: true })
  expect(first.prompts).toEqual({})
  known.set('j1', 'a fox')
  const before = listed.length
  await listWithPrompts('', none, known)
  expect(listed.length).toBe(before + 1)
  expect(listed.at(-1)?.at(-1)).toEqual({ prompts: false })
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
