import { cleanup, render } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { formatDay, formatExact, formatMtime } from '../format'
import Time from './Time.svelte'

// 2026-11-24 12:00:00 UTC, in unix seconds
const NOW = 1_795_521_600

beforeEach(() => {
  vi.useFakeTimers()
  vi.setSystemTime(NOW * 1000)
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

const time = (container: HTMLElement) =>
  container.querySelector('time') as HTMLTimeElement

it('a list time is relative with the exact time on hover', () => {
  const at = NOW - 5 * 60
  const { container } = render(Time, { props: { at, label: 'changed' } })
  expect(time(container).textContent).toBe('changed 5 min ago')
  expect(time(container).getAttribute('title')).toBe(formatExact(at))
})

it('it ages without a reload', async () => {
  const { container } = render(Time, { props: { at: NOW, label: 'changed' } })
  expect(time(container).textContent).toBe('changed just now')
  await vi.advanceTimersByTimeAsync(3 * 60 * 1000)
  expect(time(container).textContent).toBe('changed 3 min ago')
})

it('it sets datetime', () => {
  const { container } = render(Time, { props: { at: NOW } })
  expect(time(container).getAttribute('datetime')).toBe(
    '2026-11-24T12:00:00.000Z',
  )
})

it('by the agent is said', () => {
  const at = NOW - 2 * 3600
  const { container } = render(Time, {
    props: { at, label: 'changed', by: 'agent' },
  })
  expect(time(container).textContent).toBe('changed 2 h ago by the agent')
})

it('a detail time is exact, its label capitalised', () => {
  const at = NOW - 86_400 * 40
  const { container } = render(Time, {
    props: { at, mode: 'exact', label: 'created' },
  })
  expect(time(container).textContent).toBe(`Created ${formatMtime(at)}`)
  expect(time(container).getAttribute('title')).toBe(formatExact(at))
})

it('a day time keeps the date alone, and a caller title replaces the hover', () => {
  const at = NOW - 86_400 * 3
  const { container } = render(Time, {
    props: { at, mode: 'day', title: 'when a job last used it' },
  })
  expect(time(container).textContent).toBe(formatDay(at))
  expect(time(container).getAttribute('title')).toBe('when a job last used it')
})

it('an old relative time falls back to its day', () => {
  const at = NOW - 86_400 * 45
  const { container } = render(Time, { props: { at } })
  expect(time(container).textContent).toBe(formatDay(at))
})
