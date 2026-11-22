import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import JobHeader from './JobHeader.svelte'
import type { JobDetail, LedgerRequest } from '../types'

vi.mock('../api', async (original) => {
  const real = await original<typeof import('../api')>()
  return { ApiError: real.ApiError, api: {} }
})
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

afterEach(cleanup)

const job = (over: Partial<JobDetail> = {}): JobDetail =>
  ({
    id: 'j1',
    workflow: 'tiny',
    workspace: 'default',
    status: 'succeeded',
    ...over,
  }) as JobDetail

// The page reads the request and hands it down
const header = (
  j: JobDetail,
  over: { request?: LedgerRequest | null; requestGone?: boolean } = {},
) =>
  render(JobHeader, {
    ...over,
    job: j,
    jobId: j.id,
    seed: undefined,
    runVersion: undefined,
    running: false,
    cancelPending: false,
    seedVariable: null,
  })

it('links request › item to the request page, titled with the request', () => {
  header(job({ request: 'r1', item: 'hero' }), {
    request: { id: 'r1', title: 'Spring campaign' } as LedgerRequest,
  })
  const link = screen.getByRole('link', { name: 'r1 › hero' })
  expect(link.getAttribute('href')).toBe('#/ws/default/requests/r1')
  expect(link.getAttribute('title')).toBe('Spring campaign')
})

it('shows plain text, no link, once the request is gone', () => {
  header(job({ request: 'r1', item: 'hero', workspace: 'old (deleted)' }), {
    requestGone: true,
  })
  expect(screen.getByText('request no longer exists')).toBeTruthy()
  expect(screen.queryByRole('link', { name: /r1/ })).toBeNull()
  expect(screen.getByText('r1 › hero')).toBeTruthy()
})

it('links while the request is unread, and shows nothing for an unlinked job', () => {
  header(job({ request: 'r1', item: 'hero' }))
  expect(
    screen.getByRole('link', { name: 'r1 › hero' }).getAttribute('title'),
  ).toBe('request r1')
  cleanup()
  header(job())
  expect(screen.queryByText(/›/)).toBeNull()
})
