import {
  cleanup,
  fireEvent,
  render,
  screen,
  within,
} from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import RequestItem from './RequestItem.svelte'
import { planState } from './itemState'
import { attempt, doneItem, entry, item, request } from './fixtures'
import type { LedgerItem, LedgerRequest } from '../types'

vi.mock('../api', () => ({
  api: {
    galleryThumbnailUrl: (name: string, ws?: string) =>
      `/thumb/${name}?ws=${ws}`,
  },
}))
vi.mock('../confirm.svelte', () => ({ confirmDialog: vi.fn() }))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

afterEach(cleanup)

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
    onsave: vi.fn(),
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
  expect(proof.getAttribute('src')).toBe('/thumb/shot/a.png?ws=studio')
  expect(proof.closest('a')?.getAttribute('href')).toBe(
    '#/ws/studio/jobs/job-a',
  )
  expect(proof.closest('a')?.classList.contains('frame')).toBe(true)
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
  expect(container.querySelector('.proofcell .glyph')?.textContent).toBe('♪')
})

it('a ready item says ready, with its estimate and an empty slot', () => {
  const { container } = render(
    RequestItem,
    props([item('a', { ready: true, estimate_minutes: 6 })]),
  )
  expect(chip(container)?.textContent).toBe('ready')
  expect(chip(container)?.classList.contains('ready')).toBe(true)
  expect(container.querySelector('.figure')?.textContent).toBe('est. ~6 min')
  expect(container.querySelector('.proofcell .slot')).toBeTruthy()
  // the slot is never a frame: the frame is only ever the picture
  expect(container.querySelector('.proofcell .frame')).toBeNull()
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
    'shown as the result (latest)',
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
  expect(
    screen
      .getByRole('link', { name: 'open these in the gallery' })
      .getAttribute('href'),
  ).toBe('#/ws/studio/gallery?req=r1%2Fa')
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
  expect(takes[0].querySelector('.picked')).toBeTruthy()
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

it('a failed item says none of its attempts succeeded; a cancelled one is not failed', () => {
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
  expect(screen.getByText('2 attempts, none succeeded')).toBeTruthy()
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
        comment: { text: 'too long', at: '2026-01-01T00:00:00+00:00' },
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

it("links an item's jobs through the req filter", async () => {
  const { container } = render(RequestItem, props([withOutput('a', image)]))
  await fireEvent.click(title('Opening shot'))
  expect(
    within(container.querySelector('.more') as HTMLElement)
      .getByRole('link', { name: 'its jobs' })
      .getAttribute('href'),
  ).toBe('#/ws/studio/jobs?req=r1%2Fa')
})

const noted = (at: string) =>
  item('a', { title: 'Shot 3', comment: { text: 'warmer\nand slower', at } })

it('a closed row with a note shows its first line', () => {
  const { container } = render(
    RequestItem,
    props([noted('2026-01-01T00:05:00+00:00')], 'a', {
      history: [entry(9, 'agent', 'replanned')],
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
  expect(screen.getByText(/Nothing is sent to the agent/)).toBeTruthy()
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
    screen.getByRole('button', { name: 'Leave a note for the agent' }),
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

it('draws what a later increment hands it: a handle, an action, a menu', async () => {
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
  })
  expect(container.querySelector('.handle button')?.textContent).toBe('grip')
  expect(container.querySelector('.action button')?.textContent).toBe('Run…')
  expect(container.querySelector('.menu button')?.textContent).toBe('more')
})
