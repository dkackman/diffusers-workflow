import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import JobHeader from './JobHeader.svelte'
import type { JobDetail, LedgerRequest } from '../types'

const { rerunJob, cancelJob, confirmDialog, failure } = vi.hoisted(() => ({
  rerunJob: vi.fn(),
  cancelJob: vi.fn(),
  confirmDialog: vi.fn(),
  failure: vi.fn(),
}))
vi.mock('../api', async (original) => {
  const real = await original<typeof import('../api')>()
  return { ApiError: real.ApiError, api: { rerunJob, cancelJob } }
})
vi.mock('../toast', () => ({
  notify: { error: vi.fn(), success: vi.fn(), failure, undo: vi.fn() },
}))
vi.mock('../confirm.svelte', () => ({ confirmDialog }))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

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
  opts: Record<string, unknown> = {},
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
    ...opts,
  })

it('titles the version badge with its ordinal', () => {
  header(job({ run_id: 'r2' } as Partial<JobDetail>), {}, { runVersion: 2 })
  expect(screen.getByText('v2').getAttribute('title')).toBe(
    '2nd job of this workflow in this folder',
  )
})

it('titles the job with its workflow file, and the JSON id beside it when it differs', () => {
  header(job({ workflow: 'ZImage', workflow_name: 'models/z-image' }))
  expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('z-image')
  expect(screen.getByText('ZImage', { selector: '.secondary' })).toBeTruthy()
  expect(screen.getByRole('link', { name: '← Jobs' })).toHaveAttribute(
    'href',
    '#/ws/default/jobs',
  )
  cleanup()
  // a job with no file name (an inline definition) is titled by its id,
  // which is then not repeated
  header(job({ workflow: 'tiny', workflow_name: null }))
  expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('tiny')
  expect(document.querySelector('.secondary')).toBeNull()
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

it('a double click on Run again queues one job', async () => {
  let release!: (v: { id: string }) => void
  rerunJob.mockReturnValue(new Promise((r) => (release = r)))
  header(job())
  const button = screen.getByRole('button', { name: /Run again$/ })
  await fireEvent.click(button)
  await fireEvent.click(button)
  expect(rerunJob).toHaveBeenCalledTimes(1)
  expect(button.getAttribute('aria-busy')).toBe('true')
  release({ id: 'j2' })
})

const running = (startedAgo: number) =>
  header(
    job({
      status: 'running',
      workflow: 'models/e2e-card',
      started_at: Date.now() / 1000 - startedAgo,
    }),
    {},
    { running: true },
  )

it('cancel asks only past 60 seconds', async () => {
  cancelJob.mockResolvedValue({})
  running(10)
  await fireEvent.click(screen.getByRole('button', { name: /Cancel$/ }))
  await vi.waitFor(() => expect(cancelJob).toHaveBeenCalledTimes(1))
  expect(confirmDialog).not.toHaveBeenCalled()
  cleanup()
  cancelJob.mockClear()
  confirmDialog.mockResolvedValue(false)
  running(125)
  await fireEvent.click(screen.getByRole('button', { name: /Cancel$/ }))
  await vi.waitFor(() => expect(confirmDialog).toHaveBeenCalledTimes(1))
  expect(confirmDialog.mock.calls[0][0]).toMatch(
    /^Cancel e2e-card\? It has run for 2m.*; the work so far is lost\.$/,
  )
  expect(confirmDialog.mock.calls[0][1]).toMatchObject({
    confirmLabel: 'Cancel job',
    cancelLabel: 'Keep running',
    danger: true,
  })
  expect(cancelJob).not.toHaveBeenCalled()
})

it('a failed cancel is reported', async () => {
  const boom = new Error('gone')
  cancelJob.mockRejectedValue(boom)
  running(5)
  await fireEvent.click(screen.getByRole('button', { name: /Cancel$/ }))
  await vi.waitFor(() =>
    expect(failure).toHaveBeenCalledWith('cancel e2e-card', boom),
  )
})
