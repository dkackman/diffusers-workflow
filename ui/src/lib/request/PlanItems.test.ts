// The plan list on its own: which rows it draws and what it offers on them,
// open and closed. What each edit writes is the page's (RequestPage.edit).
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import type { LedgerRequest } from '../types'
import { fakeLedger } from './fakeLedger'
import { item, request } from './fixtures'
import { planState } from './itemState'
import PlanItems from './PlanItems.svelte'
import { PlanStore } from './planStore.svelte'

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())

beforeEach(() => localStorage.clear())
afterEach(cleanup)

function show(initial: LedgerRequest, over: object = {}) {
  const ledger = fakeLedger(initial)
  const store = new PlanStore({
    get: () => ledger.get(),
    put: (body) => ledger.put(body),
  })
  store.take(ledger.current)
  const handlers = {
    ongo: vi.fn(),
    onreload: vi.fn(),
    onsay: vi.fn(),
    onsee: vi.fn(),
  }
  render(PlanItems, {
    store,
    request: initial,
    plan: planState(initial),
    workspace: 'default',
    ...handlers,
    ...over,
  })
  return handlers
}

const items = [
  item('a', { ready: true }),
  item('b'),
  item('gone', { dropped: true }),
]

it('lists the live items in order, the dropped ones apart', () => {
  show(request(items))
  const plan = screen.getByRole('list', { name: 'The plan' })
  expect(plan.children).toHaveLength(2)
  expect(
    screen.getByRole('button', { name: 'Reorder Item a, position 1 of 2' }),
  ).toBeTruthy()
  expect(screen.getByTestId('dropped-group').textContent).toContain(
    'Dropped (1)',
  )
  expect(screen.getByRole('button', { name: 'Restore' })).toBeTruthy()
})

it('offers Run only where the page can run, and runs that item', async () => {
  show(request(items))
  expect(screen.queryByRole('button', { name: 'Run…' })).toBeNull()
  cleanup()
  const onrun = vi.fn()
  show(request(items), { onrun })
  await fireEvent.click(screen.getByRole('button', { name: 'Run…' }))
  expect(onrun).toHaveBeenCalledWith('a')
})

it('a closed request has no grip or Run, and offers Reopen', async () => {
  const onreopen = vi.fn()
  show(request(items, { state: 'done' }), { onrun: vi.fn(), onreopen })
  expect(screen.queryByRole('button', { name: /^Reorder / })).toBeNull()
  expect(screen.queryByRole('button', { name: 'Run…' })).toBeNull()
  await fireEvent.click(screen.getByRole('button', { name: 'Reopen' }))
  expect(onreopen).toHaveBeenCalledTimes(1)
})

it('says so when nothing is planned', () => {
  show(request([]))
  expect(screen.getByText(/has not planned any items/)).toBeTruthy()
  expect(screen.queryByRole('list', { name: 'The plan' })).toBeNull()
})

it('an arrow on a grip says where the item went', async () => {
  const { onsay } = show(request(items))
  await fireEvent.keyDown(
    screen.getByRole('button', { name: /^Reorder Item a/ }),
    { key: 'ArrowDown' },
  )
  expect(onsay).toHaveBeenCalledWith('Item a moved to position 2 of 2.')
})
