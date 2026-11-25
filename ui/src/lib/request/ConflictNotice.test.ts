import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import ConflictNotice from './ConflictNotice.svelte'
import type { Conflict } from './conflict'
import { entry } from './fixtures'

vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))
afterEach(cleanup)

const theirs = entry(14, 'agent', 'retake with warmer light')
const conflict = (over: Partial<Conflict>): Conflict => ({
  id: 1,
  kind: 'changed',
  intent: { kind: 'drop', key: 's3' },
  refusal: null,
  key: 's3',
  title: 'Shot 3',
  entry: theirs,
  changes: ['1 new take'],
  ...over,
})

it('a change that went on top is a status line that can be dismissed', async () => {
  const ondismiss = vi.fn()
  const onsee = vi.fn()
  render(ConflictNotice, {
    applied: entry(14, 'agent', 'added shot 7'),
    ondismiss,
    onsee,
  })
  const line = screen.getByRole('status')
  expect(line.textContent).toContain(
    "The agent changed the plan while you were editing (r14: 'added shot 7'). Your change was applied on top.",
  )
  expect(screen.queryByRole('alert')).toBeNull()
  await fireEvent.click(screen.getByRole('button', { name: 'See changes' }))
  expect(onsee).toHaveBeenCalledWith(14)
  await fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }))
  expect(ondismiss).toHaveBeenCalled()
})

it('says only that the plan changed when the history cannot say who', () => {
  render(ConflictNotice, { applied: null, ondismiss: vi.fn(), onsee: vi.fn() })
  expect(screen.getByRole('status').textContent).toContain(
    'The plan changed while you were editing. Your change was applied on top.',
  )
  expect(screen.queryByRole('button', { name: 'See changes' })).toBeNull()
})

it('a drop held back is an alert that asks: drop anyway, or keep it', async () => {
  const onagain = vi.fn()
  const ondismiss = vi.fn()
  render(ConflictNotice, { conflict: conflict({}), onagain, ondismiss })
  expect(screen.getByRole('alert').textContent).toContain(
    "While you were dropping this, the agent changed it (r14: 'retake with warmer light'): 1 new take.",
  )
  await fireEvent.click(screen.getByRole('button', { name: 'Drop anyway' }))
  expect(onagain).toHaveBeenCalledOnce()
  await fireEvent.click(screen.getByRole('button', { name: 'Keep it' }))
  expect(ondismiss).toHaveBeenCalledOnce()
})

it('an accept held back asks in its own words', () => {
  render(ConflictNotice, {
    conflict: conflict({ intent: { kind: 'accept', key: 's3', job: 'j1' } }),
    onagain: vi.fn(),
    ondismiss: vi.fn(),
  })
  expect(screen.getByRole('alert').textContent).toContain(
    'While you were choosing a take of this, the agent changed it',
  )
  expect(screen.getByRole('button', { name: 'Accept anyway' })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Leave it' })).toBeTruthy()
})

it('an edit that no longer applies says what was not done and why', () => {
  const gone = (refusal: Conflict['refusal'], over: Partial<Conflict> = {}) => {
    cleanup()
    render(ConflictNotice, {
      conflict: conflict({
        kind: 'gone',
        refusal,
        entry: entry(14, 'agent', 'merged into shot 2'),
        changes: [],
        ...over,
      }),
      ondismiss: vi.fn(),
    })
    return screen.getByRole('alert').textContent
  }
  expect(gone('item_gone')).toContain(
    "Not applied: the agent removed Shot 3 in r14 ('merged into shot 2').",
  )
  expect(gone('item_dropped')).toContain(
    "Not applied: the agent dropped Shot 3 in r14 ('merged into shot 2').",
  )
  expect(gone('attempt_gone')).toContain(
    "Not applied: that take is no longer one of Shot 3's takes (r14: 'merged into shot 2').",
  )
  expect(gone('anchor_gone')).toContain(
    'Not applied: the item Shot 3 was moved beside is no longer in the plan',
  )
  expect(gone('item_gone', { entry: null })).toContain(
    'Not applied: another edit removed Shot 3.',
  )
  // nothing to do anyway: it can only be dismissed
  expect(screen.queryByRole('button', { name: /anyway/ })).toBeNull()
  expect(screen.getByRole('button', { name: 'Dismiss' })).toBeTruthy()
})

it('offers no again button while the request is closed', () => {
  render(ConflictNotice, { conflict: conflict({}), ondismiss: vi.fn() })
  expect(screen.queryByRole('button', { name: 'Drop anyway' })).toBeNull()
  expect(screen.getByRole('button', { name: 'Keep it' })).toBeTruthy()
})
