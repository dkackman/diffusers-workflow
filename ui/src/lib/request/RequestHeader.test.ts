import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { createRawSnippet } from 'svelte'
import { afterEach, expect, it, vi } from 'vitest'
import RequestHeader from './RequestHeader.svelte'
import { planState } from './itemState'
import { attempt, doneItem, entry, item, request } from './fixtures'
import type { LedgerItem, LedgerRequest } from '../types'

vi.mock('../confirm.svelte', () => ({ confirmDialog: vi.fn() }))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

afterEach(cleanup)

function props(items: LedgerItem[], over: Partial<LedgerRequest> = {}) {
  const req = request(items, over)
  return {
    request: req,
    plan: planState(req),
    workspace: 'studio',
    onselect: vi.fn(),
    onsavenote: vi.fn(),
    onreload: vi.fn(),
  }
}
const running = item('shot', {
  title: 'Shot 3',
  status: 'running',
  attempts: [attempt('j', 'running')],
})
const failed = item('bad', {
  title: 'Shot 2',
  status: 'failed',
  attempts: [attempt('k', 'failed')],
})

it('leads with where the request stands and what needs the user', () => {
  const { container } = render(
    RequestHeader,
    props(
      [
        doneItem('cast', { title: 'Cast sheet' }),
        running,
        failed,
        item('next', { title: 'Shot 4', ready: true }),
        item('gone', { dropped: true, status: 'dropped' }),
      ],
      { title: 'A 90-second short about a lighthouse', id: 'lighthouse-short' },
    ),
  )
  expect(
    screen.getByRole('heading', {
      name: 'A 90-second short about a lighthouse',
    }),
  ).toBeTruthy()
  expect(container.querySelector('.tag')?.textContent).toBe('open')
  expect(screen.getByText('lighthouse-short')).toBeTruthy()
  expect(container.querySelector('.standing')?.textContent).toBe(
    'Rendering Shot 3. 1 of 4 done.',
  )
  expect(container.querySelector('.needs')?.textContent).toBe(
    'Needs you: 1 ready to run, 1 failed.',
  )
  // the revision is the engine's detail: it is in the rail, not here
  expect(container.textContent).not.toMatch(/revision/)
})

it('leaves out the needs line when nothing needs the user', () => {
  const { container } = render(RequestHeader, props([doneItem('a'), running]))
  expect(container.querySelector('.needs')).toBeNull()
})

it('draws one named segment an item, each a way to its row', async () => {
  const given = props([
    doneItem('cast', { title: 'Cast sheet' }),
    running,
    item('gone', { dropped: true, status: 'dropped' }),
  ])
  render(RequestHeader, given)
  const segments = screen.getAllByRole('button', { name: /: (done|running)$/ })
  expect(segments.map((s) => s.getAttribute('aria-label'))).toEqual([
    'Cast sheet: done',
    'Shot 3: running',
  ])
  expect(segments[1].classList.contains('running')).toBe(true)
  await fireEvent.click(segments[1])
  expect(given.onselect).toHaveBeenCalledWith('shot')
})

it('bands a plan too long for a segment an item', () => {
  const many = Array.from({ length: 41 }, (_, i) =>
    i < 30 ? doneItem(`d${i}`) : item(`p${i}`),
  )
  render(RequestHeader, props(many))
  expect(screen.queryByRole('button', { name: /: done$/ })).toBeNull()
  expect(
    screen.getByRole('img', { name: '30 done, 11 not started' }),
  ).toBeTruthy()
})

it('gives minutes spent with no tilde, what is left, and the notes waiting', () => {
  const note = { text: 'warmer', at: '2026-01-01T00:05:00+00:00' }
  const { container } = render(
    RequestHeader,
    props([item('a', { comment: note }), doneItem('b')], {
      spent_minutes: 38.2,
      remaining_minutes: 52,
      comment: note,
      history: [entry(1, 'agent', 'opened')],
    }),
  )
  expect(
    container
      .querySelector('.figures')
      ?.textContent?.replace(/\s+/g, ' ')
      .trim(),
  ).toBe('38 min spent ~52 min left 2 notes waiting')
})

it('says nothing of what is left on a closed request', () => {
  const { container } = render(
    RequestHeader,
    props([doneItem('a')], {
      state: 'done',
      spent_minutes: 4,
      remaining_minutes: 3,
      history: [entry(2, 'agent', 'all in', { state: 'done' })],
    }),
  )
  expect(container.querySelector('.figures')?.textContent?.trim()).toBe(
    '4 min spent',
  )
  expect(container.querySelector('.tag')?.classList.contains('done')).toBe(true)
  expect(container.querySelector('.standing')?.textContent).toMatch(
    /^Marked done by the agent on /,
  )
})

it('links to its jobs and outputs through the req filter', () => {
  render(RequestHeader, props([doneItem('a')]))
  expect(screen.getByText('its jobs').getAttribute('href')).toBe(
    '#/ws/studio/jobs?req=r1',
  )
  expect(screen.getByText('its outputs').getAttribute('href')).toBe(
    '#/ws/studio/gallery?req=r1',
  )
})

it("holds the note on the whole request, by an item note's rules", async () => {
  const given = props([doneItem('a')], {
    comment: { text: 'make it all warmer', at: '2026-01-01T00:05:00+00:00' },
  })
  given.onsavenote.mockResolvedValue(undefined)
  render(RequestHeader, given)
  expect(screen.getByText('make it all warmer')).toBeTruthy()
  expect(screen.getByText(/Nothing is sent to the agent/)).toBeTruthy()
  await fireEvent.click(screen.getByRole('button', { name: 'Edit note' }))
  const box = screen.getByLabelText('Note for the agent on this request')
  await fireEvent.input(box, { target: { value: 'cooler, actually' } })
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  expect(given.onsavenote).toHaveBeenCalledWith('cooler, actually')
})

it('has no filled button of its own, and a place for the one a page hands it', () => {
  const bare = render(RequestHeader, props([item('a', { ready: true })]))
  expect(bare.container.querySelector('.primary')).toBeNull()
  cleanup()
  const { container } = render(RequestHeader, {
    ...props([item('a', { ready: true })]),
    actions: createRawSnippet(() => ({
      render: () => '<button type="button">Run 1 ready…</button>',
    })),
  })
  expect(container.querySelector('.primary button')?.textContent).toBe(
    'Run 1 ready…',
  )
})
