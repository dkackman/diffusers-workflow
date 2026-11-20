import { expect, it } from 'vitest'
import { ApiError } from './apiErrors'
import { listingFailure, listingWorkspace, queueLength } from './jobsListing'
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

it("a refusal is shown as the server's own message, and empties the list", () => {
  expect(
    listingFailure(new ApiError("Request 'R 1' must match ^[a-z0-9]", 422)),
  ).toEqual({ message: "Request 'R 1' must match ^[a-z0-9]", refused: true })
  expect(listingFailure(new ApiError('Unknown workspace nope', 404))).toEqual({
    message: 'Unknown workspace nope',
    refused: true,
  })
})

it('a server fault or an unreachable server keeps the rows and says which', () => {
  expect(listingFailure(new ApiError('Internal Server Error', 500))).toEqual({
    message: 'The server could not list the jobs: Internal Server Error',
    refused: false,
  })
  expect(listingFailure(new TypeError('Failed to fetch'))).toEqual({
    message: 'Could not reach the server: Failed to fetch',
    refused: false,
  })
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
