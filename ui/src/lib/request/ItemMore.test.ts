import {
  cleanup,
  fireEvent,
  render,
  screen,
  within,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import ItemMore from './ItemMore.svelte'
import { moreId, planState } from './itemState'
import { attempt, doneItem, item, request } from './fixtures'
import type { LedgerItem, LedgerNoteRef, LedgerRequest } from '../types'

const api = vi.hoisted(() => ({
  addNote: vi.fn(),
  notesChannel: vi.fn(async () => ({ listening: false, last_seen: null })),
}))
vi.mock('../api', () => ({ api }))
vi.mock('../confirm.svelte', () => ({ confirmDialog: vi.fn() }))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

beforeEach(() => sessionStorage.clear())
afterEach(cleanup)

/** The open half of the row of `key` in a request of `items` */
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
    open: true,
    ongo: vi.fn(),
    onreload: vi.fn(),
  }
}
const more = (container: HTMLElement) =>
  container.querySelector('.more') as HTMLElement
const takes = (over: Partial<LedgerItem> = {}) =>
  doneItem('a', {
    title: 'Shot 3',
    attempts: [attempt('j1'), attempt('j2', 'failed')],
    result: { job: 'j1', run_id: 'r1', output: null },
    ...over,
  })
const add = () =>
  screen.getByRole('button', { name: 'Add a note for the agent' })
const box = () =>
  screen.getByLabelText('Note for the agent on Shot 3') as HTMLTextAreaElement

it('closed, the block is hidden, not removed: a draft outlives the close', async () => {
  const given = props([item('a', { title: 'Shot 3' })])
  const view = render(ItemMore, given)
  expect(more(view.container).id).toBe(moreId('a'))
  await fireEvent.click(add())
  await fireEvent.input(box(), { target: { value: 'half a thought' } })
  await view.rerender({ ...given, open: false })
  expect(more(view.container).hidden).toBe(true)
  expect(box().value).toBe('half a thought')
  await view.rerender({ ...given, open: true })
  expect(more(view.container).hidden).toBe(false)
  expect(box().value).toBe('half a thought')
})

it('draws the takes only while open and the item has some', async () => {
  const given = props([takes()])
  const view = render(ItemMore, given)
  expect(view.container.querySelector('.strip')).toBeTruthy()
  await view.rerender({ ...given, open: false })
  expect(view.container.querySelector('.strip')).toBeNull()
  cleanup()
  const none = render(ItemMore, props([item('a')]))
  expect(none.container.querySelector('.strip')).toBeNull()
})

it('a live item offers Accept and Note this take; a dropped one offers neither', async () => {
  const onaccept = vi.fn()
  render(ItemMore, { ...props([takes()]), onaccept })
  await fireEvent.click(
    screen.getByRole('button', { name: 'Accept take 1 for Shot 3' }),
  )
  expect(onaccept).toHaveBeenCalledWith(expect.objectContaining({ n: 1 }), 'j1')
  expect(
    screen.getByRole('button', { name: 'Note take 1 of Shot 3 for the agent' }),
  ).toBeTruthy()
  cleanup()
  render(ItemMore, {
    ...props([takes({ dropped: true, status: 'dropped' })]),
    onaccept,
  })
  expect(document.querySelector('.strip')).toBeTruthy()
  expect(screen.queryByRole('button', { name: /^Accept take/ })).toBeNull()
  expect(screen.queryByRole('button', { name: /^Note take/ })).toBeNull()
})

it('a closed request offers no Note this take', () => {
  render(ItemMore, {
    ...props([takes()], 'a', { state: 'done' }),
    onaccept: vi.fn(),
  })
  expect(screen.queryByRole('button', { name: /^Note take/ })).toBeNull()
})

it('after: and needed by: name each item with its state, and a press goes there', async () => {
  const given = props(
    [
      doneItem('cast', { title: 'Cast sheet' }),
      item('shot', { title: 'Shot 3', depends_on: ['cast'] }),
      item('cut', { title: 'The cut', depends_on: ['shot'] }),
    ],
    'shot',
  )
  const { container } = render(ItemMore, given)
  const lines = [...more(container).querySelectorAll(':scope > p.sub')]
  expect(lines.map((p) => p.textContent?.replace(/\s+/g, ' ').trim())).toEqual([
    'after: Cast sheet (done)',
    'needed by: The cut (waiting)',
  ])
  await fireEvent.click(
    within(more(container)).getByRole('button', { name: 'The cut' }),
  )
  expect(given.ongo).toHaveBeenCalledWith('cut')
})

it('a failed dependency is marked as one', () => {
  const { container } = render(
    ItemMore,
    props([
      item('shot', {
        title: 'Shot 3',
        status: 'failed',
        attempts: [attempt('j', 'failed')],
      }),
      item('cut', { title: 'The cut', depends_on: ['shot'] }),
    ]),
  )
  const line = more(container).querySelector(':scope > p.sub') as HTMLElement
  expect(line.textContent?.replace(/\s+/g, ' ').trim()).toBe(
    'after: Shot 3 (failed)',
  )
  expect(line.querySelector('.bad')?.textContent?.trim()).toBe('(failed)')
})

it('add() opens the new note’s box and focuses it, with what it was seeded with', async () => {
  const view = render(ItemMore, props([item('a', { title: 'Shot 3' })]))
  await view.component.add({
    text: 'warmer',
    refs: [{ item: 'a' }],
  })
  await vi.waitFor(() => expect(document.activeElement).toBe(box()))
  expect(box().value).toBe('warmer')
  expect(
    screen
      .getByRole('list', { name: 'What the note is about' })
      .querySelectorAll('.ref'),
  ).toHaveLength(1)
})

it('Note this take puts the take on the new note and focuses its box', async () => {
  render(ItemMore, props([takes()]))
  await fireEvent.click(
    screen.getByRole('button', { name: 'Note take 1 of Shot 3 for the agent' }),
  )
  await vi.waitFor(() => expect(document.activeElement).toBe(box()))
  const chips = screen
    .getByRole('list', { name: 'What the note is about' })
    .querySelectorAll('.ref')
  expect([...chips].map((c) => c.textContent)).toEqual(['take 1'])
})

it('the new note’s references are the bound ones, and a read keeps them', async () => {
  const take: LedgerNoteRef = { item: 'a', job: 'j1', attempt: 1 }
  const given = { ...props([takes()]), pending: [take] }
  const view = render(ItemMore, given)
  await fireEvent.click(add())
  const chips = () =>
    [
      ...screen
        .getByRole('list', { name: 'What the note is about' })
        .querySelectorAll('.ref'),
    ].map((c) => c.textContent)
  expect(chips()).toEqual(['take 1'])
  // a poll hands the half a new, equal row and request
  const again = props([takes()])
  await view.rerender({ ...again, pending: [take] })
  expect(chips()).toEqual(['take 1'])
})

it('"For the whole request instead" hands the draft and its item to the page', async () => {
  const onmove = vi.fn()
  render(ItemMore, { ...props([item('a', { title: 'Shot 3' })]), onmove })
  await fireEvent.click(add())
  await fireEvent.input(box(), { target: { value: 'all warmer' } })
  await fireEvent.click(
    screen.getByRole('button', { name: 'For the whole request instead' }),
  )
  expect(onmove).toHaveBeenCalledWith('all warmer', [{ item: 'a' }])
})

it('what it runs is one fold: the workflow and its arguments, or none', async () => {
  const { container } = render(
    ItemMore,
    props([item('a', { workflow: 'h3-shot', arguments: { seed: 7 } })]),
  )
  const fold = container.querySelector('details.runs') as HTMLDetailsElement
  expect(fold.open).toBe(false)
  expect(
    fold.querySelector('summary')?.textContent?.replace(/\s+/g, ' ').trim(),
  ).toBe('Runs h3-shot with 1 argument')
  expect(
    [...fold.querySelectorAll('.args li')].map((li) => li.textContent),
  ).toEqual(['seed = 7'])
  cleanup()
  const bare = render(ItemMore, props([item('a', { workflow: 'h3-shot' })]))
  const none = bare.container.querySelector('details.runs') as HTMLElement
  expect(
    none.querySelector('summary')?.textContent?.replace(/\s+/g, ' ').trim(),
  ).toBe('Runs h3-shot')
  expect(none.textContent).toContain('No arguments.')
  cleanup()
  const agent = render(ItemMore, props([item('a', { workflow: null })]))
  expect(agent.container.querySelector('details.runs')).toBeNull()
})
