import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { createRawSnippet } from 'svelte'
import { afterEach, expect, it, vi } from 'vitest'
import RequestHeader from './RequestHeader.svelte'
import { planState } from './itemState'
import { api } from '../test/apiMock'
import { attempt, doneItem, entry, item, request } from './fixtures'
import type { LedgerItem, LedgerRequest } from '../types'
import { formatMtime } from '../format'

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
api.addNote.mockReturnValue(undefined)
api.editNote.mockReturnValue(undefined)
api.notesChannel.mockImplementation(async () => ({
  listening: false,
  last_seen: null,
}))
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

it('pressing a needs part goes to its first item', async () => {
  const p = props([
    item('cast', { title: 'Cast sheet', ready: true }),
    failed,
    item('plate', { title: 'Plate', ready: true }),
  ])
  const { container } = render(RequestHeader, p)
  expect(container.querySelector('.needs')?.textContent).toBe(
    'Needs you: 2 ready to run, 1 failed.',
  )
  const ready = screen.getByRole('button', { name: '2 ready to run' })
  expect(ready.getAttribute('title')).toBe('Go to Cast sheet')
  expect(ready.classList.contains('bare')).toBe(true)
  await fireEvent.click(ready)
  expect(p.onselect).toHaveBeenLastCalledWith('cast')
  const bad = screen.getByRole('button', { name: '1 failed' })
  expect(bad.getAttribute('title')).toBe('Go to Shot 2')
  await fireEvent.click(bad)
  expect(p.onselect).toHaveBeenLastCalledWith('bad')
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
  const at = '2026-01-01T00:05:00+00:00'
  const note = { id: 'n1', text: 'warmer', created: at, at }
  const { container } = render(
    RequestHeader,
    props([item('a', { notes: [{ ...note, id: 'n2' }] }), doneItem('b')], {
      spent_minutes: 38.2,
      remaining_minutes: 52,
      notes: [note],
      history: [entry(1, 'agent', 'opened')],
    }),
  )
  expect(
    container
      .querySelector('.figures')
      ?.textContent?.replace(/\s+/g, ' ')
      .trim(),
  ).toBe('38 min spent about 52 min left 2 notes waiting')
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

it("holds the request's own notes, by an item note's rules", async () => {
  const at = '2026-01-01T00:05:00+00:00'
  const given = props([doneItem('a')], {
    notes: [{ id: 'n1', text: 'make it all warmer', created: at, at }],
  })
  api.editNote.mockResolvedValue({ revision: 4 })
  render(RequestHeader, given)
  expect(screen.getByText('make it all warmer')).toBeTruthy()
  expect(screen.getByText(/The agent isn't connected\. It reads/)).toBeTruthy()
  await fireEvent.click(screen.getByRole('button', { name: 'Edit note' }))
  const box = screen.getByLabelText(
    'Edit your note for the agent on this request',
  )
  await fireEvent.input(box, { target: { value: 'cooler, actually' } })
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  await waitFor(() => expect(given.onreload).toHaveBeenCalled())
  expect(api.editNote).toHaveBeenCalledWith('studio', {
    request: 'r1',
    note: 'n1',
    text: 'cooler, actually',
  })
})

it("the request's new note can be about several items, picked from the plan", async () => {
  api.addNote.mockResolvedValue({ revision: 4 })
  const given = props([
    doneItem('a', { title: 'Shot 3' }),
    item('b', { title: 'Shot 5' }),
  ])
  render(RequestHeader, given)
  await fireEvent.click(
    screen.getByRole('button', { name: 'Add a note for the agent' }),
  )
  const picker = () =>
    screen.getByLabelText('Add an item this note is about') as HTMLSelectElement
  picker().value = 'a'
  await fireEvent.change(picker())
  picker().value = 'b'
  await fireEvent.change(picker())
  // an item's chip goes to its row
  await fireEvent.click(screen.getByRole('button', { name: 'Shot 5' }))
  expect(given.onselect).toHaveBeenCalledWith('b')
  await fireEvent.input(
    screen.getByLabelText('Note for the agent on this request'),
    { target: { value: 'warmer light on both' } },
  )
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  await waitFor(() => expect(given.onreload).toHaveBeenCalled())
  expect(api.addNote).toHaveBeenCalledWith('studio', {
    request: 'r1',
    text: 'warmer light on both',
    refs: [{ item: 'a' }, { item: 'b' }],
  })
})

it('has no filled button of its own, and a place for the one a page hands it', () => {
  const bare = render(RequestHeader, props([item('a', { ready: true })]))
  expect(bare.container.querySelector('.primary')).toBeNull()
  cleanup()
  const { container } = render(RequestHeader, {
    ...props([item('a', { ready: true })]),
    actions: createRawSnippet(() => ({
      render: () => '<button type="button">Run 1…</button>',
    })),
  })
  expect(container.querySelector('.primary button')?.textContent).toBe('Run 1…')
})

const UNTIL = '2026-11-21T15:30:00+00:00'
const held = (
  key: string,
  state: 'holding' | 'redirected' | 'due',
  over: Partial<LedgerItem> = {},
) =>
  item(key, {
    title: `Shot ${key}`,
    review: { until: UNTIL, state, seconds_left: 125 },
    ...over,
  })
const banner = (container: HTMLElement) =>
  container.querySelector('.hold p')?.textContent

it('shows one hold banner, worded by its most urgent state, with Run now for the ready rows', async () => {
  const onrunheld = vi.fn()
  const { container } = render(RequestHeader, {
    ...props([
      held('a', 'holding', { ready: true }),
      held('b', 'holding'),
      held('c', 'redirected'),
    ]),
    onrunheld,
  })
  await waitFor(() =>
    expect(banner(container)).toMatch(
      /^2 items waiting for your OK\. The agent isn't connected, so nothing runs at .+ unless it reconnects\. Run them here, or leave a note\.$/,
    ),
  )
  expect(container.querySelectorAll('.hold')).toHaveLength(1)
  // not a live region: its digits would be read out every second
  expect(container.querySelector('.hold')?.closest('[aria-live]')).toBeNull()
  await fireEvent.click(screen.getByRole('button', { name: 'Run now…' }))
  expect(onrunheld).toHaveBeenCalledWith(['a'])
})

it('has no Run now when no held row is ready, and no banner with nothing held', () => {
  const { container, unmount } = render(RequestHeader, {
    ...props([held('b', 'holding')]),
    onrunheld: vi.fn(),
  })
  expect(banner(container)).toMatch(/^1 item waiting for your OK\./)
  expect(screen.queryByRole('button', { name: 'Run now…' })).toBeNull()
  unmount()
  const free = render(RequestHeader, props([item('a', { ready: true })]))
  expect(free.container.querySelector('.hold')).toBeNull()
})

it('counts down while a session listens, and says a redirect with Run now', async () => {
  api.notesChannel.mockResolvedValue({ listening: true, last_seen: null })
  const { container, unmount } = render(RequestHeader, {
    ...props([held('a', 'holding', { ready: true, estimate_minutes: 3 })]),
    onrunheld: vi.fn(),
  })
  await waitFor(() =>
    expect(banner(container)).toMatch(
      /^1 item waiting for your OK \(about 3 min by the agent's estimate\)\. The agent will run it at .+ if it's connected \(in 02:05\), unless you leave a note\.$/,
    ),
  )
  expect(screen.getByRole('button', { name: 'Run now…' })).toBeTruthy()
  unmount()
  const redirected = render(RequestHeader, {
    ...props([held('a', 'redirected', { ready: true })]),
    onrunheld: vi.fn(),
  })
  expect(banner(redirected.container)).toBe(
    "You left a note after these were held, so they won't start on their own. The agent will read it and may revise them; run them here if they're right as they are.",
  )
  expect(screen.getByRole('button', { name: 'Run now…' })).toBeTruthy()
  redirected.unmount()
  api.notesChannel.mockResolvedValue({ listening: false, last_seen: null })
})

it("says the agent's latest word under what needs the user, and nothing of it once closed", () => {
  const history = [
    entry(1, 'agent', 'opened'),
    entry(2, 'user', 'note added'),
    entry(3, 'agent', 'board-01 now blue; awaiting go-ahead to draw'),
  ]
  const view = render(RequestHeader, props([item('a')], { history }))
  const said = view.container.querySelector('.said') as HTMLElement
  expect(said.textContent).toContain('The agent, r3')
  expect(said.textContent).toContain('awaiting go-ahead to draw')
  cleanup()
  const closed = render(
    RequestHeader,
    props([item('a')], { history, state: 'done' }),
  )
  expect(closed.container.querySelector('.said')).toBeNull()
})

it("the agent's latest word carries its time, and hovering it gives the whole word", () => {
  const why = 'board-01 now blue; awaiting go-ahead to draw'
  const history = [entry(1, 'agent', 'opened'), entry(3, 'agent', why)]
  const view = render(RequestHeader, props([item('a')], { history }))
  const time = view.container.querySelector('.said time') as HTMLElement
  expect(view.container.querySelector('.said .who')?.textContent).toMatch(
    new RegExp(
      `^\\s*The agent, r3 ·\\s+${formatMtime(Date.parse(history[1].at) / 1000)}:\\s*$`,
    ),
  )
  expect(time.getAttribute('title')).toBe(why)
})

it("the agent's latest word still shows when its stored time does not parse", () => {
  const history = [entry(2, 'agent', 'all set', { at: 'not a time' })]
  const view = render(RequestHeader, props([item('a')], { history }))
  const said = view.container.querySelector('.said') as HTMLElement
  expect(said.textContent).toMatch(/The agent, r2 ·\s+unknown time:/)
  expect(said.textContent).toContain('all set')
  expect(said.querySelector('[datetime]')).toBeNull()
})
