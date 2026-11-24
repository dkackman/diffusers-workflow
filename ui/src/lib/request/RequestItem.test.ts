import {
  cleanup,
  fireEvent,
  render,
  screen,
  within,
} from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import RequestItem from './RequestItem.svelte'
import source from './RequestItem.svelte?raw'
import { planState } from './itemState'
import { attempt, doneItem, entry, item, request } from './fixtures'
import type { LedgerItem, LedgerRequest } from '../types'

const api = vi.hoisted(() => ({
  addNote: vi.fn(),
  notesChannel: vi.fn(async () => ({ listening: false, last_seen: null })),
}))
vi.mock('../api', () => ({
  api: {
    ...api,
  },
}))
vi.mock('../confirm.svelte', () => ({ confirmDialog: vi.fn() }))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

// A draft is kept for the tab (draft.ts): none carries into the next test
afterEach(() => {
  cleanup()
  sessionStorage.clear()
})

let ids = 0
const note = (text: string, at: string) => ({
  id: `n${++ids}`,
  text,
  created: at,
  at,
})

/** The props for the row of `key` in a request of `items` */
function props(
  items: LedgerItem[],
  key = items[items.length - 1].key,
  over: Partial<LedgerRequest> = {},
) {
  const req = request(items, over)
  const row = planState(req).items.find((s) => s.item.key === key)
  if (!row) throw new Error(`no item ${key}`)
  return {
    row,
    request: req,
    workspace: 'studio',
    onreload: vi.fn(),
    ongo: vi.fn(),
  }
}
const chip = (container: HTMLElement) => container.querySelector('.row .chip')
const title = (name: string) => screen.getByRole('button', { name })
const image = { name: 'shot/a.png', kind: 'image' as const }
const withOutput = (key: string, output: typeof image | null) =>
  doneItem(key, {
    title: 'Opening shot',
    attempts: [attempt(`job-${key}`, 'succeeded', { output })],
    result: { job: `job-${key}`, run_id: `run-${key}`, output },
    spent_minutes: 6.25,
  })

it('a done item shows its result as the proof, linked to the job', () => {
  const { container } = render(RequestItem, props([withOutput('a', image)]))
  const proof = screen.getByRole('img', { name: 'result of Opening shot' })
  expect(proof.getAttribute('src')).toBe(
    '/api/gallery/shot/a.png/thumbnail?workspace=studio',
  )
  expect(proof.closest('a')?.getAttribute('href')).toBe(
    '#/ws/studio/jobs/job-a',
  )
  expect(proof.closest('a')?.classList.contains('filetile')).toBe(true)
  expect(proof.closest('a')?.classList.contains('tile-strip')).toBe(true)
  expect(chip(container)?.textContent).toBe('done')
  expect(chip(container)?.classList.contains('succeeded')).toBe(true)
  // measured minutes, so no tilde
  expect(container.querySelector('.figure')?.textContent).toBe('6.3 min')
  expect(title('Opening shot').getAttribute('aria-expanded')).toBe('false')
})

it('marks a video proof as one, and stands a sound by a glyph', () => {
  const video = { name: 'shot/clip.mp4', kind: 'video' as const }
  const view = render(RequestItem, props([withOutput('a', video as never)]))
  expect(view.container.querySelector('.proofcell .videobadge')).toBeTruthy()
  cleanup()
  const sound = { name: 'shot/a.wav', kind: 'audio' as const }
  const { container } = render(
    RequestItem,
    props([withOutput('a', sound as never)]),
  )
  expect(container.querySelector('.proofcell img')).toBeNull()
  expect(
    container.querySelector('.proofcell .tile-glyph')?.textContent?.trim(),
  ).toBe('♪')
  // named where it stands alone
  expect(
    screen.getByRole('link', { name: 'result of Opening shot' }),
  ).toBeTruthy()
})

it('a ready item says ready, with its estimate and an empty slot', () => {
  const { container } = render(
    RequestItem,
    props([item('a', { ready: true, estimate_minutes: 6 })]),
  )
  expect(chip(container)?.textContent).toBe('ready')
  expect(chip(container)?.classList.contains('ready')).toBe(true)
  expect(container.querySelector('.figure')?.textContent).toBe(
    'est. about 6 min',
  )
  expect(container.querySelector('.proofcell .slot')).toBeTruthy()
  // the slot is never a tile: the tile is only ever the file
  expect(container.querySelector('.proofcell .filetile')).toBeNull()
  // no control is drawn that this increment cannot act on
  expect(screen.queryByRole('button', { name: /run/i })).toBeNull()
})

it('a waiting item names what it waits on, each a way to that row', async () => {
  const given = props([
    item('shot', {
      title: 'Shot 3',
      status: 'failed',
      attempts: [attempt('j', 'failed')],
    }),
    item('score', { title: 'Score', workflow: null }),
    item('cut', { title: 'The cut', depends_on: ['shot', 'score'] }),
  ])
  const { container } = render(RequestItem, given)
  expect(chip(container)?.textContent).toBe('waiting')
  const line = container.querySelector('.main .sub') as HTMLElement
  expect(line.textContent?.replace(/\s+/g, ' ')).toContain(
    'waits on Shot 3 (failed), Score (planned)',
  )
  expect(line.querySelector('.bad')?.textContent?.trim()).toBe('(failed)')
  await fireEvent.click(within(line).getByRole('button', { name: 'Shot 3' }))
  expect(given.ongo).toHaveBeenCalledWith('shot')
})

it('a failed row that cannot run again yet says what it waits on', () => {
  const failed = {
    status: 'failed' as const,
    attempts: [attempt('j', 'failed')],
  }
  const { container } = render(
    RequestItem,
    props([
      item('plate', { title: 'Background plate', ...failed }),
      item('comp', { title: 'Composite', depends_on: ['plate'], ...failed }),
    ]),
  )
  expect(container.querySelector('.main .sub')?.textContent).toBe(
    'Failed. It can run again once Background plate is done.',
  )
})

it('says only the agent can run an item with no stored workflow', () => {
  const { container } = render(
    RequestItem,
    props([item('a', { workflow: null })]),
  )
  expect(chip(container)?.textContent).toBe('planned')
  expect(
    screen.getByText('No stored workflow. Only the agent can run this.'),
  ).toBeTruthy()
})

it('a running item is the live row, open, with a way to its job', () => {
  const { container } = render(
    RequestItem,
    props([
      item('a', {
        title: 'Shot 3',
        status: 'running',
        attempts: [attempt('job-now', 'running', { minutes: 2 })],
      }),
    ]),
  )
  expect(container.querySelector('.row')?.classList.contains('row-live')).toBe(
    true,
  )
  expect(chip(container)?.textContent).toBe('running')
  expect(container.querySelector('.figure')?.textContent).toBe('2 min so far')
  expect(
    screen.getByRole('link', { name: 'Open job' }).getAttribute('href'),
  ).toBe('#/ws/studio/jobs/job-now')
  expect(title('Shot 3').getAttribute('aria-expanded')).toBe('true')
  expect(container.querySelector('.proofcell .pulse-dot')).toBeTruthy()
})

const twoTakes = (over: Partial<LedgerItem> = {}) =>
  doneItem('a', {
    title: 'Cast sheet',
    attempts: [
      attempt('j1', 'succeeded', { output: image, note: 'first pass' }),
      attempt('j2', 'failed'),
      attempt('j3', 'succeeded', {
        output: { name: 'shot/c.png', kind: 'image' },
      }),
    ],
    result: {
      job: 'j3',
      run_id: 'r3',
      output: { name: 'shot/c.png', kind: 'image' },
    },
    ...over,
  })

it('with several takes and none accepted, opens and marks the latest as the result', () => {
  const { container } = render(RequestItem, props([twoTakes()]))
  expect(title('Cast sheet').getAttribute('aria-expanded')).toBe('true')
  expect(
    container.querySelector('.main')?.textContent?.replace(/\s+/g, ' '),
  ).toContain('3 takes latest, none accepted')
  const takes = [...container.querySelectorAll('.strip .tile')]
  expect(takes.map((t) => t.querySelector('.n')?.textContent)).toEqual([
    'take 1',
    'take 2',
    'take 3',
  ])
  expect(takes[2].querySelector('.mark')?.textContent?.trim()).toBe(
    'latest take',
  )
  expect(takes[0].querySelector('.mark')).toBeNull()
  // a take that made nothing is a slot with its status in words
  expect(takes[1].querySelector('.slot')).toBeTruthy()
  expect(takes[1].querySelector('.chip')?.textContent).toBe('failed')
  expect(takes[0].querySelector('.note')?.textContent).toBe('first pass')
  expect(takes[0].querySelector('a')?.getAttribute('href')).toBe(
    '#/ws/studio/jobs/j1',
  )
  // read-only: there is nothing to accept with yet
  expect(screen.queryByRole('button', { name: /accept/i })).toBeNull()
})

it('marks the accepted take, and stays closed', async () => {
  const { container } = render(
    RequestItem,
    props([
      twoTakes({
        accepted: { job: 'j1', run_id: 'r1' },
        result: { job: 'j1', run_id: 'r1', output: image },
      }),
    ]),
  )
  expect(title('Cast sheet').getAttribute('aria-expanded')).toBe('false')
  expect(container.querySelector('.main .mark')?.textContent?.trim()).toBe(
    'accepted',
  )
  await fireEvent.click(title('Cast sheet'))
  const takes = [...container.querySelectorAll('.strip .tile')]
  expect(takes[0].querySelector('.mark')?.textContent?.trim()).toBe('accepted')
  expect(takes[0].querySelector('.filetile.picked.tile-take')).toBeTruthy()
  expect(takes[2].querySelector('.mark')).toBeNull()
})

it("says when the accepted take's files are gone", () => {
  render(
    RequestItem,
    props([
      twoTakes({
        accepted: { job: 'j1', run_id: 'r1' },
        accepted_missing: true,
      }),
    ]),
  )
  expect(screen.getByText(/The accepted take's files are gone/)).toBeTruthy()
})

it('a failed item says none of its takes succeeded; a cancelled one is not failed', () => {
  const failed = render(
    RequestItem,
    props([
      item('a', {
        status: 'failed',
        attempts: [attempt('j1', 'failed'), attempt('j2', 'cancelled')],
      }),
    ]),
  )
  expect(chip(failed.container)?.classList.contains('failed')).toBe(true)
  expect(screen.getByText('2 takes, none succeeded')).toBeTruthy()
  cleanup()
  const { container } = render(
    RequestItem,
    props([
      item('a', { status: 'failed', attempts: [attempt('j1', 'cancelled')] }),
    ]),
  )
  expect(chip(container)?.textContent).toBe('cancelled')
  expect(chip(container)?.classList.contains('failed')).toBe(false)
})

it('a dropped item is struck, keeps its note, and offers no way to edit one', async () => {
  const { container } = render(
    RequestItem,
    props([
      item('a', {
        title: 'Cut scene',
        dropped: true,
        status: 'dropped',
        notes: [note('too long', '2026-01-01T00:00:00+00:00')],
      }),
    ]),
  )
  expect(container.querySelector('li')?.classList.contains('dropped')).toBe(
    true,
  )
  expect(chip(container)?.textContent).toBe('dropped')
  await fireEvent.click(title('Cut scene'))
  expect(
    within(container.querySelector('.more') as HTMLElement).getByText(
      'too long',
    ),
  ).toBeTruthy()
  expect(screen.queryByRole('button', { name: /note/i })).toBeNull()
  expect(screen.queryByRole('textbox')).toBeNull()
})

it('opened, says what it follows, what needs it and what it would run', async () => {
  const given = props(
    [
      doneItem('cast', { title: 'Cast sheet' }),
      item('shot', {
        title: 'Shot 3',
        depends_on: ['cast'],
        ready: true,
        workflow: 'h3-shot',
        arguments: { prompt: 'a lighthouse', seed: 7 },
      }),
      item('cut', { title: 'The cut', depends_on: ['shot'] }),
    ],
    'shot',
  )
  const { container } = render(RequestItem, given)
  const more = container.querySelector('.more') as HTMLElement
  expect(more.hidden).toBe(true)
  await fireEvent.click(title('Shot 3'))
  expect(more.hidden).toBe(false)
  expect(more.id).toBe(title('Shot 3').getAttribute('aria-controls'))
  const text = more.textContent?.replace(/\s+/g, ' ') ?? ''
  expect(text).toContain('after: Cast sheet (done)')
  expect(text).toContain('needed by: The cut (waiting)')
  expect(text).toContain('Runs h3-shot with 2 arguments')
  expect(
    [...more.querySelectorAll('.args li')].map((li) => li.textContent),
  ).toEqual(['prompt = a lighthouse', 'seed = 7'])
  await fireEvent.click(within(more).getByRole('button', { name: 'The cut' }))
  expect(given.ongo).toHaveBeenCalledWith('cut')
})

it('an open item shows its takes once: the row picture steps aside, and the jobs and outputs links are in its menu', async () => {
  const { container } = render(RequestItem, props([withOutput('a', image)]))
  await fireEvent.click(title('Opening shot'))
  expect(container.querySelector('.proofcell')?.classList).toContain('under')
  const more = within(container.querySelector('.more') as HTMLElement)
  expect(more.queryByRole('link', { name: 'its jobs' })).toBeNull()
  expect(more.queryByRole('link', { name: /gallery/ })).toBeNull()
})

const noted = (at: string) =>
  item('a', { title: 'Shot 3', notes: [note('warmer\nand slower', at)] })

it('a closed row with notes shows the first line of the latest, and how many more', () => {
  const at = '2026-01-01T00:05:00+00:00'
  const { container } = render(
    RequestItem,
    props(
      [
        item('a', {
          title: 'Shot 3',
          notes: [note('first', at), note('second\nmore', at)],
        }),
      ],
      'a',
      { agent_wrote_at: '2026-01-01T00:09:00+00:00' },
    ),
  )
  expect(
    container.querySelector('.peek')?.textContent?.replace(/\s+/g, ' ').trim(),
  ).toBe('second +1')
})

it('a closed row with a note shows its first line', () => {
  const { container } = render(
    RequestItem,
    props([noted('2026-01-01T00:05:00+00:00')], 'a', {
      history: [entry(9, 'agent', 'replanned')],
      agent_wrote_at: entry(9, 'agent', 'replanned').at,
    }),
  )
  expect(title('Shot 3').getAttribute('aria-expanded')).toBe('false')
  expect(container.querySelector('.peek')?.textContent?.trim()).toBe('warmer')
})

it('opens a row whose note the agent has had no chance to read', () => {
  const { container } = render(
    RequestItem,
    props([noted('2026-01-01T00:05:00+00:00')]),
  )
  expect(title('Shot 3').getAttribute('aria-expanded')).toBe('true')
  expect(container.querySelector('.peek')).toBeNull()
  expect(screen.getByText(/The agent isn't connected\. It reads/)).toBeTruthy()
})

it("the user's toggle outlives a refresh, and closing a row keeps its draft", async () => {
  const running = (minutes: number) =>
    props([
      item('a', {
        title: 'Shot 3',
        status: 'running',
        attempts: [attempt('j', 'running', { minutes })],
      }),
    ])
  const view = render(RequestItem, running(1))
  await fireEvent.click(
    screen.getByRole('button', { name: 'Add a note for the agent' }),
  )
  const box = () =>
    view.container.querySelector('textarea') as HTMLTextAreaElement
  await fireEvent.input(box(), { target: { value: 'half a thought' } })
  await fireEvent.click(title('Shot 3'))
  expect(title('Shot 3').getAttribute('aria-expanded')).toBe('false')
  // a poll hands the row a new, equal item: it would open by default
  await view.rerender(running(2))
  expect(title('Shot 3').getAttribute('aria-expanded')).toBe('false')
  expect(view.container.querySelector('.figure')?.textContent).toBe(
    '2 min so far',
  )
  await fireEvent.click(title('Shot 3'))
  expect(box().value).toBe('half a thought')
})

it('draws the controls a page hands it: a handle, an action, a menu, a notice', async () => {
  const { createRawSnippet } = await import('svelte')
  const slot = (label: string) =>
    createRawSnippet(() => ({
      render: () => `<button type="button">${label}</button>`,
    }))
  const { container } = render(RequestItem, {
    ...props([item('a', { ready: true })]),
    handle: slot('grip'),
    action: slot('Run…'),
    menu: slot('more'),
    notice: slot('asked'),
  })
  expect(container.querySelector('li > button')?.textContent).toBe('asked')
  expect(container.querySelector('.handle button')?.textContent).toBe('grip')
  expect(container.querySelector('.action button')?.textContent).toBe('Run…')
  expect(container.querySelector('.menu button')?.textContent).toBe('more')
})

const goodTakes = (over: Partial<LedgerItem> = {}) =>
  doneItem('a', {
    title: 'Shot 3',
    attempts: [attempt('j1'), attempt('j2'), attempt('j3', 'failed')],
    result: { job: 'j2', run_id: 'r2', output: null },
    ...over,
  })

it('each good take can be accepted, and says which one is', async () => {
  const onaccept = vi.fn()
  render(RequestItem, { ...props([goodTakes()]), onaccept })
  const buttons = screen.getAllByRole('button', { name: /^Accept take/ })
  // the failed take has no button
  expect(buttons.map((b) => b.getAttribute('aria-label'))).toEqual([
    'Accept take 1 for Shot 3',
    'Accept take 2 for Shot 3',
  ])
  expect(buttons.map((b) => b.getAttribute('aria-pressed'))).toEqual([
    'false',
    'false',
  ])
  expect(buttons.map((b) => b.textContent)).toEqual(['Accept', 'Accept'])
  await fireEvent.click(buttons[0])
  expect(onaccept).toHaveBeenCalledWith(expect.objectContaining({ n: 1 }), 'j1')
})

it('pressing the accepted take clears the choice', async () => {
  const onaccept = vi.fn()
  render(RequestItem, {
    ...props([goodTakes({ accepted: { job: 'j1', run_id: 'r1' } })]),
    onaccept,
  })
  await fireEvent.click(title('Shot 3'))
  const accepted = screen.getByRole('button', {
    name: 'Accept take 1 for Shot 3',
  })
  expect(accepted.getAttribute('aria-pressed')).toBe('true')
  expect(accepted.textContent).toBe('Accepted')
  await fireEvent.click(accepted)
  expect(onaccept).toHaveBeenCalledWith(expect.objectContaining({ n: 1 }), null)
})

it("with the accepted take's files gone, says to accept another", () => {
  const { container } = render(RequestItem, {
    ...props([
      goodTakes({
        accepted: { job: 'j1', run_id: 'r1' },
        accepted_missing: true,
      }),
    ]),
    onaccept: vi.fn(),
  })
  expect(container.querySelector('.gone')?.textContent).toContain(
    "The accepted take's files are gone. Showing the latest take instead. Accept another to settle it.",
  )
})

it('a dropped item offers no accept', async () => {
  render(RequestItem, {
    ...props([goodTakes({ dropped: true, status: 'dropped' })]),
    onaccept: vi.fn(),
  })
  await fireEvent.click(title('Shot 3'))
  expect(screen.queryByRole('button', { name: /^Accept take/ })).toBeNull()
})

it('the menu can ask the row to open its note box', async () => {
  const { createRawSnippet } = await import('svelte')
  let api: { note: () => void } | undefined
  render(RequestItem, {
    ...props([item('a', { title: 'Shot 4', ready: true })]),
    menu: createRawSnippet<[unknown, { note: () => void }]>((_row, given) => {
      api = given()
      return { render: () => '<span></span>' }
    }),
  })
  expect(title('Shot 4').getAttribute('aria-expanded')).toBe('false')
  api?.note()
  const box = await screen.findByLabelText('Note for the agent on Shot 4')
  expect(title('Shot 4').getAttribute('aria-expanded')).toBe('true')
  await vi.waitFor(() => expect(document.activeElement).toBe(box))
})

it('"Note this take" adds the take and its file to the new note, once', async () => {
  api.addNote.mockResolvedValue({ revision: 4 })
  const proof = { name: 'run/shot3-take1.png', kind: 'image' as const }
  const given = props([
    goodTakes({
      attempts: [
        attempt('j1', 'succeeded', { output: proof }),
        attempt('j2'),
        attempt('j3', 'failed'),
      ],
    }),
  ])
  render(RequestItem, given)
  await fireEvent.click(
    screen.getByRole('button', { name: 'Note take 1 of Shot 3 for the agent' }),
  )
  const box = await screen.findByLabelText('Note for the agent on Shot 3')
  await vi.waitFor(() => expect(document.activeElement).toBe(box))
  await fireEvent.click(
    screen.getByRole('button', { name: 'Note take 1 of Shot 3 for the agent' }),
  )
  await fireEvent.click(
    screen.getByRole('button', { name: 'Note take 3 of Shot 3 for the agent' }),
  )
  const chips = screen
    .getByRole('list', { name: 'What the note is about' })
    .querySelectorAll('.ref')
  expect([...chips].map((c) => c.textContent)).toEqual([
    'take 1 · shot3-take1.png',
    'take 3',
  ])
  await fireEvent.input(box, { target: { value: 'take 1, warmer' } })
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  await vi.waitFor(() => expect(given.onreload).toHaveBeenCalled())
  expect(api.addNote.mock.calls[0][1]).toEqual({
    request: 'r1',
    item: 'a',
    text: 'take 1, warmer',
    refs: [
      { item: 'a', job: 'j1', output: 'run/shot3-take1.png' },
      { item: 'a', job: 'j3' },
    ],
  })
})

// Open by default for its takes, then the agent accepts one: by default
// it would close. A note being drafted keeps it open
const accepted = () => goodTakes({ accepted: { job: 'j1', run_id: 'r1' } })

it('"Note this take" keeps the row open when a read would close it', async () => {
  const view = render(RequestItem, props([goodTakes()]))
  expect(title('Shot 3').getAttribute('aria-expanded')).toBe('true')
  await fireEvent.click(
    screen.getByRole('button', { name: 'Note take 1 of Shot 3 for the agent' }),
  )
  await view.rerender(props([accepted()]))
  expect(title('Shot 3').getAttribute('aria-expanded')).toBe('true')
  expect(screen.getByLabelText('Note for the agent on Shot 3')).toBeTruthy()
})

it("the box's own Add a note keeps the row open while it holds a draft", async () => {
  const view = render(RequestItem, props([goodTakes()]))
  expect(title('Shot 3').getAttribute('aria-expanded')).toBe('true')
  await fireEvent.click(
    screen.getByRole('button', { name: 'Add a note for the agent' }),
  )
  const box = await screen.findByLabelText('Note for the agent on Shot 3')
  await fireEvent.input(box, { target: { value: 'half a thought' } })
  await view.rerender(props([accepted()]))
  expect(title('Shot 3').getAttribute('aria-expanded')).toBe('true')
  expect(
    (
      screen.getByLabelText(
        'Note for the agent on Shot 3',
      ) as HTMLTextAreaElement
    ).value,
  ).toBe('half a thought')
})

it("the menu's note keeps the row open when a read would close it", async () => {
  const { createRawSnippet } = await import('svelte')
  let api: { note: () => void } | undefined
  const menu = createRawSnippet<[unknown, { note: () => void }]>(
    (_row, given) => {
      api = given()
      return { render: () => '<span></span>' }
    },
  )
  const view = render(RequestItem, { ...props([goodTakes()]), menu })
  api?.note()
  await screen.findByLabelText('Note for the agent on Shot 3')
  await view.rerender({ ...props([accepted()]), menu })
  expect(title('Shot 3').getAttribute('aria-expanded')).toBe('true')
})

it('a saved note keeps the row open: the read brings it back waiting', async () => {
  api.addNote.mockResolvedValue({ revision: 4 })
  const given = props([goodTakes()])
  const view = render(RequestItem, given)
  await fireEvent.click(
    screen.getByRole('button', { name: 'Add a note for the agent' }),
  )
  const box = await screen.findByLabelText('Note for the agent on Shot 3')
  await fireEvent.input(box, { target: { value: 'warmer' } })
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  await vi.waitFor(() => expect(given.onreload).toHaveBeenCalled())
  const at = '2026-01-01T00:05:00+00:00'
  await view.rerender(
    props([goodTakes({ ...accepted(), notes: [note('warmer', at)] })]),
  )
  expect(title('Shot 3').getAttribute('aria-expanded')).toBe('true')
})

it('a dropped item offers no "Note this take"', async () => {
  render(RequestItem, props([goodTakes({ dropped: true, status: 'dropped' })]))
  await fireEvent.click(title('Shot 3'))
  expect(screen.queryByRole('button', { name: /^Note take/ })).toBeNull()
})

it('marks a held row with a chip: its countdown, a note left, or time up', () => {
  const hold = (state: 'holding' | 'redirected' | 'due') =>
    item('a', {
      ready: true,
      review: { until: '2026-11-21T12:00:00+00:00', state, seconds_left: 125 },
    })
  const chip = (state: 'holding' | 'redirected' | 'due') => {
    const { container, unmount } = render(RequestItem, props([hold(state)]))
    const text = container.querySelector('.chip.held')?.textContent
    unmount()
    return text
  }
  expect(chip('holding')).toBe('waiting for your OK · 02:05')
  expect(chip('redirected')).toBe('on hold: note left')
  expect(chip('due')).toBe('review time up')
  const { container } = render(RequestItem, props([item('a')]))
  expect(container.querySelector('.chip.held')).toBeNull()
})

const proofOf = (n: number) => ({
  name: `run/take${n}.png`,
  kind: 'image' as const,
})
const filedTakes = (over: Partial<LedgerItem> = {}) =>
  doneItem('a', {
    title: 'Shot 3',
    attempts: [
      attempt('j1', 'succeeded', { output: proofOf(1) }),
      attempt('j2', 'failed'),
      attempt('j3', 'succeeded', { output: proofOf(3) }),
    ],
    result: { job: 'j3', run_id: 'r3', output: proofOf(3) },
    ...over,
  })

it('a take opens in the viewer, and Accept there accepts it', async () => {
  const onaccept = vi.fn()
  render(RequestItem, { ...props([filedTakes()]), onaccept })
  await fireEvent.click(
    screen.getByRole('button', { name: 'take 1 of Shot 3' }),
  )
  const viewer = screen.getByRole('dialog')
  expect(
    within(viewer)
      .getByRole('link', { name: 'Open the job' })
      .getAttribute('href'),
  ).toBe('#/ws/studio/jobs/j1')
  await fireEvent.click(within(viewer).getByRole('button', { name: 'Accept' }))
  expect(onaccept).toHaveBeenCalledWith(expect.objectContaining({ n: 1 }), 'j1')
  expect(location.hash).toContain('open=')
})

it('the caption of a take links its job', () => {
  render(RequestItem, props([filedTakes()]))
  expect(
    screen.getByRole('link', { name: 'take 3' }).getAttribute('href'),
  ).toBe('#/ws/studio/jobs/j3')
})

it('Compare takes opens the newest four', () => {
  const attempts = [1, 2, 3, 4, 5].map((n) =>
    attempt(`j${n}`, 'succeeded', { output: proofOf(n) }),
  )
  render(RequestItem, props([filedTakes({ attempts })]))
  const href = screen
    .getByRole('link', { name: 'Compare takes' })
    .getAttribute('href')
  expect(href).toBe(
    '#/ws/studio/compare/' +
      [2, 3, 4, 5].map((n) => encodeURIComponent(`run/take${n}.png`)).join('/'),
  )
})

it('a take with no file stays a slot and does not open', () => {
  render(RequestItem, props([filedTakes()]))
  expect(screen.queryByRole('button', { name: 'take 2 of Shot 3' })).toBeNull()
  expect(
    screen.getAllByRole('button', { name: /^take \d of Shot 3$/ }),
  ).toHaveLength(2)
})

// A retry that writes the same file: two takes, one output name
const sameName = { name: 'run/out.png', kind: 'image' as const }
const retried = () =>
  filedTakes({
    attempts: [
      attempt('j1', 'succeeded', { output: sameName }),
      attempt('j2', 'succeeded', { output: sameName }),
    ],
    result: { job: 'j2', run_id: 'r2', output: sameName },
  })

it('takes that share an output name are told apart: the second accepts as the second', async () => {
  const onaccept = vi.fn()
  render(RequestItem, { ...props([retried()]), onaccept })
  await fireEvent.click(
    screen.getByRole('button', { name: 'take 2 of Shot 3' }),
  )
  const viewer = screen.getByRole('dialog')
  expect(
    within(viewer)
      .getByRole('link', { name: 'Open the job' })
      .getAttribute('href'),
  ).toBe('#/ws/studio/jobs/j2')
  await fireEvent.click(within(viewer).getByRole('button', { name: 'Accept' }))
  expect(onaccept).toHaveBeenCalledWith(expect.objectContaining({ n: 2 }), 'j2')
})

it('previous and next visit both takes that share a name', async () => {
  render(RequestItem, props([retried()]))
  await fireEvent.click(
    screen.getByRole('button', { name: 'take 1 of Shot 3' }),
  )
  const viewer = screen.getByRole('dialog')
  const jobLink = () =>
    within(viewer)
      .getByRole('link', { name: 'Open the job' })
      .getAttribute('href')
  expect(jobLink()).toBe('#/ws/studio/jobs/j1')
  await fireEvent.click(within(viewer).getByRole('button', { name: /next/i }))
  expect(jobLink()).toBe('#/ws/studio/jobs/j2')
  await fireEvent.click(within(viewer).getByRole('button', { name: /prev/i }))
  expect(jobLink()).toBe('#/ws/studio/jobs/j1')
})

it('Compare takes names a file once, though two takes wrote it', () => {
  const attempts = [
    attempt('j1', 'succeeded', { output: proofOf(1) }),
    attempt('j2', 'succeeded', { output: sameName }),
    attempt('j3', 'succeeded', { output: sameName }),
  ]
  render(RequestItem, props([filedTakes({ attempts })]))
  expect(
    screen.getByRole('link', { name: 'Compare takes' }).getAttribute('href'),
  ).toBe(
    '#/ws/studio/compare/' +
      [proofOf(1).name, sameName.name].map(encodeURIComponent).join('/'),
  )
})

it('a failed row says why, collapsed or open, under what it waits on', () => {
  const failed = item('a', {
    title: 'Plate',
    status: 'failed',
    attempts: [attempt('j1', 'failed', { reason: 'e2e: this attempt failed' })],
  })
  const { container } = render(RequestItem, props([failed]))
  expect(container.textContent).toContain(
    'Take 1 failed: e2e: this attempt failed',
  )
})

it("an open row's proof cell takes no room", async () => {
  // not `a`: a test above leaves a deep link to one of a's takes in the hash
  const { container } = render(RequestItem, props([withOutput('p', image)]))
  const row = container.querySelector('.row') as HTMLElement
  expect(row.classList.contains('under')).toBe(false)
  await fireEvent.click(title('Opening shot'))
  // the takes below show the picture: its track is 0 and the title spans it
  expect(row.classList.contains('under')).toBe(true)
  expect(source).toMatch(/\.row\.under\s*\{\s*--proof:\s*0px;?\s*\}/)
  expect(source).toMatch(/\.proofcell\.under\s*\{\s*display:\s*none;?\s*\}/)
  // open with no takes, the slot keeps its place
  cleanup()
  const plain = render(RequestItem, props([item('b', { ready: true })]))
  await fireEvent.click(title('Item b'))
  expect(plain.container.querySelector('.row.under')).toBeNull()
})

it('under 900px of panel, the chip and figure go under the title', () => {
  const narrow = source.match(
    /@container \(max-width: 900px\)\s*\{([\s\S]*?)\n {2}\}/,
  )
  expect(narrow?.[1]).toMatch(/'handle proof main main tail'/)
  expect(narrow?.[1]).toMatch(/'handle proof chip figure tail'/)
})

it('the bound references of a new note outlive a read', async () => {
  const given = props([goodTakes()])
  const view = render(RequestItem, given)
  await fireEvent.click(
    screen.getByRole('button', { name: 'Note take 1 of Shot 3 for the agent' }),
  )
  await view.rerender(props([goodTakes()]))
  await fireEvent.click(
    screen.getByRole('button', { name: 'Note take 2 of Shot 3 for the agent' }),
  )
  const chips = screen
    .getByRole('list', { name: 'What the note is about' })
    .querySelectorAll('.ref')
  expect([...chips].map((c) => c.textContent)).toEqual(['take 1', 'take 2'])
})
