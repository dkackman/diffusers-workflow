// One row of the Jobs list: what it says about a job in each state, and
// which moves it offers on a queued one.
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import type { JobSummary } from '../types'
import JobRow from './JobRow.svelte'

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())

afterEach(cleanup)

const job = (over: Partial<JobSummary> = {}) =>
  ({
    id: 'j-1',
    status: 'queued',
    workflow: 'shot',
    workflow_name: 'shot',
    workspace: 'film',
    created_at: 1_700_000_000,
    started_at: null,
    finished_at: null,
    ...over,
  }) as JobSummary

function show(over: Partial<JobSummary> = {}, props: object = {}) {
  const onmove = vi.fn()
  render(JobRow, {
    job: job(over),
    queueSize: 3,
    showWorkspace: false,
    prompt: '',
    percent: null,
    proof: null,
    onmove,
    ...props,
  })
  return onmove
}
const row = () => screen.getByRole('link')
const moves = () =>
  screen
    .queryAllByRole('button', { name: /^move j-1 / })
    .map((b) => b.getAttribute('aria-label'))

it('links to the job in its workspace, under its id', () => {
  show({}, { prompt: 'a cat\non a mat', showWorkspace: true })
  expect(row().getAttribute('href')).toBe('#/ws/film/jobs/j-1')
  expect(row().textContent).toContain('j-1')
  expect(row().textContent).toContain('a cat')
  expect(screen.getByTitle('workspace').textContent?.trim()).toBe('film')
})

it.each([
  [0, ['move j-1 down one place']],
  [1, ['to the front of the queue', 'up one place', 'down one place']],
  [2, ['to the front of the queue', 'up one place']],
])('queued at %i offers only the moves that go somewhere', (at, expected) => {
  show({ queue_position: at })
  expect(moves()).toHaveLength(expected.length)
  expected.forEach((words, i) => expect(moves()[i]).toContain(words))
})

it('hands up a move, and says every button waits while one is in flight', async () => {
  const onmove = show({ queue_position: 1 })
  await fireEvent.click(screen.getByRole('button', { name: /front/ }))
  expect(onmove).toHaveBeenCalledWith(expect.anything(), 'j-1', 'front')
  cleanup()
  show({ queue_position: 1 }, { busyId: 'other' })
  for (const button of screen.getAllByRole('button'))
    expect(button.getAttribute('aria-disabled')).toBe('true')
})

it('offers no move when the job is alone in the queue', () => {
  show({ queue_position: 0 }, { queueSize: 1 })
  expect(moves()).toEqual([])
  expect(row().textContent).toContain('#1')
})

it('a running job shows how far it is, a finished one how long it took', () => {
  show({ status: 'running', started_at: 1_700_000_000 }, { percent: 40 })
  expect(row().textContent).toContain('40%')
  cleanup()
  show({
    status: 'succeeded',
    started_at: 1_700_000_000,
    finished_at: 1_700_000_090,
  })
  expect(row().textContent).toContain('1m 30s')
  expect(row().textContent).not.toContain('%')
})
