import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import JobHeader from './JobHeader.svelte'
import { ApiError, api } from '../api'
import type { JobDetail } from '../types'

vi.mock('../api', async (original) => {
  const real = await original<typeof import('../api')>()
  return { ApiError: real.ApiError, api: { getRequest: vi.fn() } }
})
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

afterEach(() => {
  cleanup()
  vi.mocked(api.getRequest).mockReset()
})

const job = (over: Partial<JobDetail> = {}): JobDetail =>
  ({
    id: 'j1',
    workflow: 'tiny',
    workspace: 'default',
    status: 'succeeded',
    ...over,
  }) as JobDetail

const header = (j: JobDetail) =>
  render(JobHeader, {
    job: j,
    jobId: j.id,
    seed: undefined,
    runVersion: undefined,
    running: false,
    cancelPending: false,
    seedVariable: null,
  })

it('links request › item to the request page, titled with the request', async () => {
  vi.mocked(api.getRequest).mockResolvedValue({
    id: 'r1',
    title: 'Spring campaign',
  } as never)
  header(job({ request: 'r1', item: 'hero' }))
  const link = screen.getByRole('link', { name: 'r1 › hero' })
  expect(link.getAttribute('href')).toBe('#/ws/default/requests/r1')
  await waitFor(() =>
    expect(link.getAttribute('title')).toBe('Spring campaign'),
  )
  expect(api.getRequest).toHaveBeenCalledWith('default', 'r1')
})

it('shows plain text, no link, once the request is gone', async () => {
  vi.mocked(api.getRequest).mockRejectedValue(new ApiError('gone', 404))
  header(job({ request: 'r1', item: 'hero', workspace: 'old (deleted)' }))
  await waitFor(() =>
    expect(screen.getByText('request no longer exists')).toBeTruthy(),
  )
  expect(screen.queryByRole('link', { name: /r1/ })).toBeNull()
  expect(screen.getByText('r1 › hero')).toBeTruthy()
})

it('keeps the link on other errors, and shows nothing for an unlinked job', async () => {
  vi.mocked(api.getRequest).mockRejectedValue(new ApiError('boom', 500))
  header(job({ request: 'r1', item: 'hero' }))
  await waitFor(() => expect(api.getRequest).toHaveBeenCalled())
  expect(screen.getByRole('link', { name: 'r1 › hero' })).toBeTruthy()
  cleanup()
  header(job())
  expect(screen.queryByText(/›/)).toBeNull()
})
