import { afterEach, expect, it } from 'vitest'
import { cleanup, render, waitFor } from '@testing-library/svelte'
import StatusPopover from './StatusPopover.svelte'
import type { HealthInfo } from './types'

afterEach(() => {
  cleanup()
  document.body.innerHTML = ''
})

const health: HealthInfo = {
  status: 'ok',
  version: '1',
  hostname: 'lem',
  device: 'cuda',
  mcp: false,
  queued: 0,
  worker_alive: true,
  current_job: 'job-a',
  workers: [
    {
      device: 'cuda:0',
      name: 'cuda:0 NVIDIA GeForce RTX 3090',
      vram_gb: 24,
      current_job: 'job-a',
      alive: true,
    },
    {
      device: 'cuda:1',
      name: 'cuda:1 NVIDIA GeForce RTX 3090',
      vram_gb: 24,
      current_job: 'job-b',
      alive: true,
    },
  ],
}

it('lists every worker with a link to its current job', async () => {
  const anchor = document.createElement('button')
  document.body.appendChild(anchor)
  render(StatusPopover, { props: { open: true, anchor, health, memory: null } })
  const pop = await waitFor(() => {
    const el = document.querySelector<HTMLElement>('[data-popover-content]')
    expect(el).not.toBeNull()
    return el!
  })
  expect(pop.textContent).toContain('cuda:0 NVIDIA GeForce RTX 3090')
  expect(pop.textContent).toContain('cuda:1 NVIDIA GeForce RTX 3090')
  const hrefs = [...pop.querySelectorAll('a')].map((a) =>
    a.getAttribute('href'),
  )
  expect(hrefs).toEqual(['#/jobs/job-a', '#/jobs/job-b'])
})
