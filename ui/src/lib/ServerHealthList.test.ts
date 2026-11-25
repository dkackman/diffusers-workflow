import { afterEach, beforeEach, expect, it } from 'vitest'
import { cleanup, render } from '@testing-library/svelte'
import ServerHealthList from './ServerHealthList.svelte'
import { connection } from './connection.svelte'
import type { HealthInfo } from './types'

const live: HealthInfo = {
  status: 'ok',
  version: '1.2',
  hostname: 'lem',
  device: 'cuda',
  mcp: false,
  queued: 3,
  worker_alive: true,
  current_job: 'job-a',
  workers: [
    {
      device: 'cuda:0',
      name: 'cuda:0 RTX 3090',
      vram_gb: 24,
      current_job: 'job-a',
      alive: true,
    },
  ],
}

beforeEach(() => {
  connection.state = 'unknown'
  connection.health = null
  connection.memory = null
})
afterEach(cleanup)

const text = (c: Element) => (c.textContent ?? '').replace(/\s+/g, ' ')

it('unreachable is red and says nothing about the worker', () => {
  connection.state = 'offline'
  const { container } = render(ServerHealthList)
  const bad = container.querySelector('.bad')
  expect(bad?.textContent).toBe('unreachable')
  expect(text(container)).not.toContain('not started')
})

it('locked names the token', () => {
  connection.state = 'locked'
  const { container } = render(ServerHealthList)
  expect(text(container)).toContain('needs the API token')
  expect(text(container)).not.toContain('unreachable')
  expect(text(container)).not.toContain('not started')
})

it('a live reading shows the worker and the queue', () => {
  connection.state = 'ok'
  connection.health = live
  const { container } = render(ServerHealthList)
  const t = text(container)
  expect(t).toContain('ok · v1.2')
  expect(t).toContain('cuda:0 RTX 3090')
  expect(t).toContain('3 queued')
})

it('says not started only for a live reading with no worker', () => {
  connection.state = 'ok'
  connection.health = { ...live, worker_alive: false, workers: [] }
  const { container } = render(ServerHealthList)
  expect(text(container)).toContain('not started')
})
