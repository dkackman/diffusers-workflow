import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import JobsPage from './JobsPage.svelte'
import { formatExact } from '../format'
import type { JobSummary } from '../types'

const data = vi.hoisted(() => ({ jobs: [] as unknown[] }))
vi.mock('../api', () => ({
  api: {
    listJobs: () =>
      Promise.resolve({ jobs: [...data.jobs], total: data.jobs.length }),
    listWorkspaces: () =>
      Promise.resolve({ default: 'default', workspaces: [] }),
  },
}))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

afterEach(cleanup)

const summary = (over: Partial<JobSummary>): JobSummary => ({
  id: 'j',
  workflow: 'w',
  workflow_name: null,
  status: 'succeeded',
  created_at: 1,
  started_at: 1,
  finished_at: 2,
  workspace: 'default',
  run_id: null,
  run_version: null,
  acknowledged: 'none',
  ...over,
})

// A history row written before the workflow was stored has a null
// workflow; the list must still render it, and every other row
it('lists a job recorded without a workflow name', async () => {
  data.jobs = [
    summary({ id: 'old-1', workflow: null, historical: true }),
    summary({ id: 'new-1', workflow: 'text-to-image' }),
  ]
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('text-to-image')).toBeTruthy())
  expect(screen.getByText(/old-1/)).toBeTruthy()
})

it('shows the card a job ran on, and nothing for one with no device', async () => {
  data.jobs = [
    summary({
      id: 'g1',
      workflow: 'on-gpu',
      device: 'cuda:1 NVIDIA GeForce RTX 3090',
    }),
    summary({ id: 'g2', workflow: 'no-gpu', device: null }),
  ]
  const { container } = render(JobsPage)
  await waitFor(() => expect(screen.getByText('no-gpu')).toBeTruthy())
  expect(screen.getByText('cuda:1 NVIDIA GeForce RTX 3090')).toBeTruthy()
  expect(container.textContent).not.toContain('cuda:0')
  expect(
    container.querySelectorAll('[title="the card this job ran on"]'),
  ).toHaveLength(1)
})

it('marks a rerun', async () => {
  data.jobs = [
    summary({ id: 'r1', workflow: 'a-rerun', rerun_of: 'abc' }),
    summary({ id: 'r2', workflow: 'fresh' }),
  ]
  const { container } = render(JobsPage)
  await waitFor(() => expect(screen.getByText('fresh')).toBeTruthy())
  expect(screen.getByTitle('a rerun of abc')).toBeTruthy()
  expect(container.querySelectorAll('[title^="a rerun of"]')).toHaveLength(1)
})

it('shows a queued job as waiting and a started one with its wait', async () => {
  data.jobs = [
    summary({ id: 'q1', workflow: 'wq', status: 'queued', started_at: null }),
    summary({ id: 's1', workflow: 'ws', created_at: 100, started_at: 160 }),
  ]
  const { container } = render(JobsPage)
  await waitFor(() => expect(screen.getByText('ws')).toBeTruthy())
  expect(container.textContent).toContain('waiting')
  expect(
    container.querySelector('[title$="waited 1m 00s in the queue"]'),
  ).toBeTruthy()
})

it('hovering a start time gives it to the second, with the finish', async () => {
  data.jobs = [
    summary({
      id: 's1',
      workflow: 'ws',
      started_at: 1700000000,
      finished_at: 1700000243,
    }),
  ]
  const { container } = render(JobsPage)
  await waitFor(() => expect(screen.getByText('ws')).toBeTruthy())
  const title = container.querySelector('.started')!.getAttribute('title')!
  expect(title).toContain(`started ${formatExact(1700000000)}`)
  expect(title).toContain(`finished ${formatExact(1700000243)}`)
})

it('reads its filters from the link and writes them back to it', async () => {
  location.hash = '#/ws/default/jobs?status=failed'
  data.jobs = [
    summary({ id: 'f1', workflow: 'broke', status: 'failed' }),
    summary({ id: 'ok', workflow: 'fine' }),
  ]
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('broke')).toBeTruthy())
  expect(screen.queryByText('fine')).toBeNull()
  const input = screen.getByPlaceholderText('filter by workflow…')
  await fireEvent.input(input, { target: { value: 'zzz' } })
  expect(location.hash).toBe('#/ws/default/jobs?status=failed&q=zzz')
  await fireEvent.click(screen.getByText('clear filters'))
  expect(location.hash).toBe('#/ws/default/jobs')
  expect(screen.getByText('fine')).toBeTruthy()
  location.hash = ''
})
