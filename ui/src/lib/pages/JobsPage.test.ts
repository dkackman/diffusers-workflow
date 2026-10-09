import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import JobsPage from './JobsPage.svelte'
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
