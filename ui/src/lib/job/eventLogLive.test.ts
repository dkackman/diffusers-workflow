import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { tick } from 'svelte'
import { api } from '../api'
import type { EventLogReply, JobEvent } from '../types'
import EventLogHarness from './EventLogHarness.svelte'

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  api: { eventLog: vi.fn() },
}))

// A server whose replies the test releases one at a time
const served: JobEvent[] = []
let held: (() => void)[] = []
let hold = false

function reply(after: number): EventLogReply {
  const events = served.filter((e) => e.seq > after)
  return {
    id: 'j1',
    status: 'running',
    events,
    last_seq: events.length ? events[events.length - 1].seq : after,
    truncated: false,
    note: null,
  }
}

beforeEach(() => {
  vi.useFakeTimers()
  served.length = 0
  held = []
  hold = false
  served.push({ seq: 0, event: 'log', message: 'line 0' })
  vi.mocked(api.eventLog).mockImplementation(async (_id, q = {}) => {
    const after = q.after ?? -1
    if (!hold || after < 0) return reply(after)
    // Reads like a request in flight: the reply is what the server held
    // when the request was sent
    const out = reply(after)
    await new Promise<void>((resolve) => held.push(resolve))
    return out
  })
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.mocked(api.eventLog).mockReset()
})

const add = (seq: number) =>
  served.push({ seq, event: 'log', message: `line ${seq}` })
const calls = () => vi.mocked(api.eventLog).mock.calls.length
const seqs = (c: HTMLElement) =>
  [...c.querySelectorAll('tbody tr')].map(
    (r) => r.querySelector('td')?.textContent,
  )
const flush = async () => {
  for (let i = 0; i < 6; i++) await vi.advanceTimersByTimeAsync(0)
  await tick()
}

async function opened(props: { live: boolean; pulse?: number }) {
  const view = render(EventLogHarness, {
    start: { pulse: 0, ...props },
  })
  await fireEvent.click(screen.getByRole('button', { name: /Event log/ }))
  await flush()
  const rerender = async (next: Parameters<typeof view.component.set>[0]) => {
    view.component.set(next)
    await tick()
  }
  return { ...view, rerender }
}

describe('EventLog live reads', () => {
  it('a beat during a read gives exactly one more read, rows once', async () => {
    const { rerender, container } = await opened({ live: true })
    expect(calls()).toBe(1)
    hold = true
    add(1)
    await vi.advanceTimersByTimeAsync(1000)
    await rerender({ live: true, pulse: 1 })
    await flush()
    expect(calls()).toBe(2)
    add(2)
    await rerender({ live: true, pulse: 2 })
    await rerender({ live: true, pulse: 3 })
    await flush()
    expect(calls()).toBe(2)
    hold = false
    held.splice(0).forEach((release) => release())
    await vi.advanceTimersByTimeAsync(1000)
    await flush()
    expect(calls()).toBe(3)
    expect(seqs(container)).toEqual(['0', '1', '2'])
  })

  it('collects the closing events when the job ends during a read', async () => {
    const { rerender, container } = await opened({ live: true })
    hold = true
    add(1)
    await vi.advanceTimersByTimeAsync(1000)
    await rerender({ live: true, pulse: 1 })
    await flush()
    expect(calls()).toBe(2)
    // The closing events arrive, then the job ends, with the read still out
    add(2)
    await rerender({ live: false, pulse: 2 })
    hold = false
    held.splice(0).forEach((release) => release())
    await flush()
    expect(calls()).toBe(3)
    expect(seqs(container)).toEqual(['0', '1', '2'])
  })

  it('reads at most twice for ten beats inside half a second', async () => {
    const { rerender } = await opened({ live: true })
    await vi.advanceTimersByTimeAsync(1000)
    const before = calls()
    for (let i = 1; i <= 10; i++) {
      add(i)
      await rerender({ live: true, pulse: i })
      await vi.advanceTimersByTimeAsync(50)
    }
    await vi.advanceTimersByTimeAsync(2000)
    await flush()
    expect(calls() - before).toBeLessThanOrEqual(2)
    expect(calls() - before).toBeGreaterThanOrEqual(1)
  })

  it('a job that ends reads at once, not after the gap', async () => {
    const { rerender, container } = await opened({ live: true })
    add(1)
    await rerender({ live: true, pulse: 1 })
    add(2)
    await rerender({ live: true, pulse: 2 })
    await rerender({ live: false, pulse: 2 })
    await flush()
    expect(seqs(container)).toEqual(['0', '1', '2'])
  })

  it('starts no read after unmount, in flight or waiting', async () => {
    const view = await opened({ live: true })
    hold = true
    add(1)
    await vi.advanceTimersByTimeAsync(1000)
    await view.rerender({ live: true, pulse: 1 })
    await flush()
    const during = calls()
    await view.rerender({ live: true, pulse: 2 })
    await view.rerender({ live: true, pulse: 3 })
    view.unmount()
    held.splice(0).forEach((release) => release())
    await vi.advanceTimersByTimeAsync(5000)
    await flush()
    expect(calls()).toBe(during)
  })

  it('a waiting read is dropped when the job changes', async () => {
    const view = await opened({ live: true })
    add(1)
    await view.rerender({ live: true, pulse: 1 })
    const before = calls()
    await view.rerender({ jobId: 'j2' })
    await vi.advanceTimersByTimeAsync(5000)
    expect(calls()).toBe(before)
  })

  it('a closed log makes no reads on beats', async () => {
    const { rerender } = await opened({ live: true })
    await fireEvent.click(screen.getByRole('button', { name: /Event log/ }))
    const before = calls()
    for (let i = 1; i <= 5; i++) await rerender({ live: true, pulse: i })
    await vi.advanceTimersByTimeAsync(5000)
    expect(calls()).toBe(before)
  })

  it('Follow stays off when a job goes live after the user turned it off', async () => {
    const { rerender } = await opened({ live: false })
    const follow = screen.getByRole('button', { name: 'Follow new events' })
    await fireEvent.click(follow) // off -> on
    await fireEvent.click(follow) // on -> off, by the user
    expect(follow.getAttribute('aria-pressed')).toBe('false')
    await rerender({ live: true, pulse: 0 })
    await flush()
    expect(follow.getAttribute('aria-pressed')).toBe('false')
  })
})
