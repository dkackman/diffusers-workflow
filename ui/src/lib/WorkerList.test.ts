import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it } from 'vitest'
import WorkerList from './WorkerList.svelte'
import type { WorkerInfo } from './types'

afterEach(cleanup)

const worker = (over: Partial<WorkerInfo>): WorkerInfo =>
  ({
    device: 'cuda:0',
    name: 'RTX',
    vram_gb: 24,
    current_job: 'j1',
    alive: true,
    ...over,
  }) as WorkerInfo

it("a worker's job links its own workspace", () => {
  render(WorkerList, { workers: [worker({ workspace: 'other' })] })
  expect(screen.getByRole('link').getAttribute('href')).toBe(
    '#/ws/other/jobs/j1',
  )
})
