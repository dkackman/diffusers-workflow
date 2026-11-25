import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { ChangeFeed, CHANGES_POLL_MS } from './changes.svelte'
import type { WorkspaceChanges } from './types'
import { api } from './test/apiMock'

vi.mock('./api', async () => (await import('./test/apiMock')).apiModule())

const snapshot = (
  kinds: Partial<WorkspaceChanges['kinds']> = {},
  { epoch = 'e1', workspaces = 0 } = {},
): WorkspaceChanges => ({
  epoch,
  workspaces,
  kinds: {
    requests: 0,
    prompts: 0,
    workflows: 0,
    assets: 0,
    gallery: 0,
    jobs: 0,
    models: 0,
    loras: 0,
    ...kinds,
  },
})

beforeEach(() => {
  vi.useFakeTimers()
  api.workspaceChanges.mockReset()
  api.workspaceChanges.mockResolvedValue(snapshot())
})
// Every watch a test makes is stopped after it, or its timer reads on
const stops: (() => void)[] = []
const watch = (feed: ChangeFeed, ...args: Parameters<ChangeFeed['watch']>) => {
  const stop = feed.watch(...args)
  stops.push(stop)
  return stop
}
afterEach(() => {
  stops.splice(0).forEach((stop) => stop())
  vi.useRealTimers()
})

const tick = () => vi.advanceTimersByTimeAsync(CHANGES_POLL_MS)

it('reads while something watches, and stops when nothing does', async () => {
  const feed = new ChangeFeed('studio')
  await vi.advanceTimersByTimeAsync(CHANGES_POLL_MS * 2)
  expect(api.workspaceChanges).not.toHaveBeenCalled()
  const a = watch(feed, ['requests'], () => {})
  const b = watch(feed, ['prompts'], () => {})
  await vi.advanceTimersByTimeAsync(0)
  // one shared read, whatever the number of holders
  expect(api.workspaceChanges).toHaveBeenCalledTimes(1)
  expect(api.workspaceChanges).toHaveBeenCalledWith('studio')
  await tick()
  expect(api.workspaceChanges).toHaveBeenCalledTimes(2)
  a()
  a() // stopping twice counts once
  await tick()
  expect(api.workspaceChanges).toHaveBeenCalledTimes(3)
  b()
  expect(feed.watching).toBe(0)
  await vi.advanceTimersByTimeAsync(CHANGES_POLL_MS * 3)
  expect(api.workspaceChanges).toHaveBeenCalledTimes(3)
})

it('calls a watcher when its kind moves, and not when another does', async () => {
  const feed = new ChangeFeed('studio')
  const onRequests = vi.fn()
  const onPrompts = vi.fn()
  watch(feed, ['requests'], onRequests)
  watch(feed, ['prompts', 'workflows'], onPrompts)
  await vi.advanceTimersByTimeAsync(0)
  // the first read is the baseline
  expect(onRequests).not.toHaveBeenCalled()
  api.workspaceChanges.mockResolvedValue(snapshot({ requests: 1 }))
  await tick()
  expect(onRequests).toHaveBeenCalledTimes(1)
  expect(onPrompts).not.toHaveBeenCalled()
  // an unmoved read calls no one
  await tick()
  expect(onRequests).toHaveBeenCalledTimes(1)
  api.workspaceChanges.mockResolvedValue(
    snapshot({ requests: 1, workflows: 4 }),
  )
  await tick()
  expect(onPrompts).toHaveBeenCalledTimes(1)
  expect(onRequests).toHaveBeenCalledTimes(1)
})

it('the server-wide workspaces counter is a kind to watch', async () => {
  const feed = new ChangeFeed('studio')
  const fn = vi.fn()
  watch(feed, ['workspaces'], fn)
  await vi.advanceTimersByTimeAsync(0)
  api.workspaceChanges.mockResolvedValue(snapshot({}, { workspaces: 2 }))
  await tick()
  expect(fn).toHaveBeenCalledTimes(1)
})

it('a new epoch calls every watcher: everything changed', async () => {
  const feed = new ChangeFeed('studio')
  const fns = [vi.fn(), vi.fn()]
  watch(feed, ['requests'], fns[0])
  watch(feed, ['gallery'], fns[1])
  await vi.advanceTimersByTimeAsync(0)
  // a restart puts the counters back at 0: only the epoch says so
  api.workspaceChanges.mockResolvedValue(snapshot({}, { epoch: 'e2' }))
  await tick()
  fns.forEach((fn) => expect(fn).toHaveBeenCalledTimes(1))
})

it('a failed read keeps the counters, so the next good one catches up', async () => {
  const feed = new ChangeFeed('studio')
  const fn = vi.fn()
  watch(feed, ['requests'], fn)
  await vi.advanceTimersByTimeAsync(0)
  api.workspaceChanges.mockRejectedValueOnce(new Error('503'))
  await tick()
  expect(fn).not.toHaveBeenCalled()
  api.workspaceChanges.mockResolvedValue(snapshot({ requests: 3 }))
  await tick()
  expect(fn).toHaveBeenCalledTimes(1)
})

it('a reply overtaken by a later read is dropped', async () => {
  const feed = new ChangeFeed('studio')
  const fn = vi.fn()
  await feed.refresh() // the baseline
  let answerSlow: (value: WorkspaceChanges) => void = () => {}
  api.workspaceChanges.mockImplementationOnce(
    () => new Promise((resolve) => (answerSlow = resolve)),
  )
  watch(feed, ['requests'], fn)
  // the watch's own first read is the slow one; a later read answers first
  api.workspaceChanges.mockResolvedValueOnce(snapshot({ requests: 2 }))
  await feed.refresh()
  expect(fn).toHaveBeenCalledTimes(1)
  // the old reply lands after it, with an older count: dropped, or the
  // next read would see `requests` move again
  answerSlow(snapshot({ requests: 1 }))
  await vi.advanceTimersByTimeAsync(0)
  api.workspaceChanges.mockResolvedValue(snapshot({ requests: 2 }))
  await tick()
  expect(fn).toHaveBeenCalledTimes(1)
})

it('does not read while the tab is hidden, and reads once on its return', async () => {
  const setHidden = (hidden: boolean) => {
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      get: () => (hidden ? 'hidden' : 'visible'),
    })
    document.dispatchEvent(new Event('visibilitychange'))
  }
  const feed = new ChangeFeed('studio')
  const stop = watch(feed, ['requests'], () => {})
  await vi.advanceTimersByTimeAsync(0)
  expect(api.workspaceChanges).toHaveBeenCalledTimes(1)
  setHidden(true)
  await vi.advanceTimersByTimeAsync(CHANGES_POLL_MS * 4)
  expect(api.workspaceChanges).toHaveBeenCalledTimes(1)
  setHidden(false)
  await vi.advanceTimersByTimeAsync(0)
  expect(api.workspaceChanges).toHaveBeenCalledTimes(2)
  stop()
})
