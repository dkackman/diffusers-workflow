import { beforeEach, expect, it, vi } from 'vitest'

const { failure, undo, success, error, confirmDialog } = vi.hoisted(() => ({
  failure: vi.fn(),
  undo: vi.fn(),
  success: vi.fn(),
  error: vi.fn(),
  confirmDialog: vi.fn(),
}))
vi.mock('./toast', () => ({ notify: { failure, undo, success, error } }))
vi.mock('./confirm.svelte', () => ({ confirmDialog }))

import { action } from './action.svelte'

beforeEach(() => vi.clearAllMocks())

it('refuses re-entry while busy', async () => {
  let release!: (v: number) => void
  const fn = vi.fn(() => new Promise<number>((r) => (release = r)))
  const act = action(fn, { doing: 'do it' })
  const first = act.run()
  const second = await act.run()
  expect(second).toBeUndefined()
  expect(act.busy).toBe(true)
  release(7)
  expect(await first).toBe(7)
  expect(act.busy).toBe(false)
  expect(fn).toHaveBeenCalledTimes(1)
})

it('re-enables and reports after a rejection', async () => {
  const boom = new Error('nope')
  const act = action(() => Promise.reject(boom), { doing: 'do it' })
  expect(await act.run()).toBeUndefined()
  expect(act.busy).toBe(false)
  expect(failure).toHaveBeenCalledWith('do it', boom)
})

it('a declined confirm never calls fn', async () => {
  confirmDialog.mockResolvedValue(false)
  const fn = vi.fn(async () => 1)
  const act = action(fn, {
    doing: 'do it',
    confirm: () => ({ message: 'Sure?', confirmLabel: 'Yes', danger: true }),
  })
  expect(await act.run()).toBeUndefined()
  expect(fn).not.toHaveBeenCalled()
  expect(act.busy).toBe(false)
  expect(confirmDialog).toHaveBeenCalledWith('Sure?', {
    confirmLabel: 'Yes',
    cancelLabel: undefined,
    danger: true,
    title: undefined,
  })
})

it('a null confirm runs without asking', async () => {
  const act = action(async () => 1, { doing: 'x', confirm: () => null })
  expect(await act.run()).toBe(1)
  expect(confirmDialog).not.toHaveBeenCalled()
})

it('a fn that answers null was declined: nothing is announced', async () => {
  const act = action(async (): Promise<{ name: string } | null> => null, {
    doing: 'x',
    success: (r) => `Saved ${r.name}`,
    undo: () => ({ message: 'm', revert: async () => {} }),
  })
  expect(await act.run()).toBeNull()
  expect(success).not.toHaveBeenCalled()
  expect(undo).not.toHaveBeenCalled()
  expect(failure).not.toHaveBeenCalled()
})

it('doing can read the arguments', async () => {
  const act = action(
    vi.fn<(n: number) => Promise<void>>().mockRejectedValue(new Error('x')),
    {
      doing: (n: number) => `cancel ${n}`,
    },
  )
  await act.run(4)
  expect(failure).toHaveBeenCalledWith('cancel 4', expect.any(Error))
})

it('success and undo toasts', async () => {
  const revert = vi.fn(async () => undefined)
  const act = action(async () => 5, {
    doing: 'x',
    success: (r) => `made ${r}`,
    undo: (r) => ({ message: `undo ${r}`, revert }),
  })
  await act.run()
  expect(success).not.toHaveBeenCalled()
  expect(undo).toHaveBeenCalledTimes(1)
  expect(undo.mock.calls[0][0]).toBe('undo 5')
  undo.mock.calls[0][1]()
  await vi.waitFor(() => expect(revert).toHaveBeenCalled())

  const plain = action(async () => 5, { doing: 'x', success: 'done' })
  await plain.run()
  expect(success).toHaveBeenCalledWith('done')
})

it('a failed revert is reported', async () => {
  const bad = new Error('revert failed')
  const act = action(async () => 1, {
    doing: 'save',
    undo: () => ({ message: 'm', revert: () => Promise.reject(bad) }),
  })
  await act.run()
  undo.mock.calls[0][1]()
  // One copy for every failed undo, whatever the action was
  await vi.waitFor(() => expect(failure).toHaveBeenCalledWith('undo that', bad))
})

it('an undo runs once, and in turn with the action it takes back', async () => {
  let finish: () => void = () => {}
  const fn = vi.fn(async (n: number) => n)
  const revert = vi.fn(
    () => new Promise<void>((resolve) => (finish = () => resolve())),
  )
  const act = action(fn, {
    doing: 'star that file',
    undo: () => ({ message: 'm', revert }),
  })
  await act.run(1)
  // A double click on Undo: the faded toast's button takes the second
  undo.mock.calls[0][1]()
  undo.mock.calls[0][1]()
  await vi.waitFor(() => expect(revert).toHaveBeenCalledTimes(1))
  // While the revert is out, the action is busy: a click does nothing
  expect(act.busy).toBe(true)
  expect(await act.run(2)).toBeUndefined()
  expect(fn).toHaveBeenCalledTimes(1)
  finish()
  await vi.waitFor(() => expect(act.busy).toBe(false))
  expect(revert).toHaveBeenCalledTimes(1)
})

it('an undo pressed while the action is busy waits for it', async () => {
  let finish: () => void = () => {}
  let calls = 0
  const fn = vi.fn(async () => {
    if (++calls === 2) await new Promise<void>((r) => (finish = () => r()))
  })
  const revert = vi.fn(async () => undefined)
  const act = action(fn, {
    doing: 'x',
    undo: () => ({ message: 'm', revert }),
  })
  await act.run()
  const second = act.run()
  undo.mock.calls[0][1]()
  await new Promise((r) => setTimeout(r, 0))
  expect(revert).not.toHaveBeenCalled()
  finish()
  await second
  await vi.waitFor(() => expect(revert).toHaveBeenCalledTimes(1))
})

it('an undo that waited re-checks its target', async () => {
  let finish: () => void = () => {}
  let calls = 0
  const fn = vi.fn(async () => {
    if (++calls === 2) await new Promise<void>((r) => (finish = () => r()))
  })
  const revert = vi.fn(async () => undefined)
  let open = 'shot-1'
  const act = action(fn, {
    doing: 'x',
    undo: () => ({
      message: 'm',
      revert,
      target: { now: () => open, away: 'That request is no longer open.' },
    }),
  })
  await act.run()
  const second = act.run()
  // Pressed while the second run is out, on the request it was made on
  undo.mock.calls[0][1]()
  await new Promise((r) => setTimeout(r, 0))
  // Then another request opens before the run finishes
  open = 'shot-2'
  finish()
  await second
  await vi.waitFor(() =>
    expect(error).toHaveBeenCalledWith('That request is no longer open.'),
  )
  expect(revert).not.toHaveBeenCalled()
  expect(act.busy).toBe(false)
})
