import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { flushSync } from 'svelte'
import { feed } from './test/feedMock'
import {
  CHANGED_MARK_MS,
  resource,
  type Resource,
  type ResourceOptions,
} from './resource.svelte'
import { workspace } from './workspaceState.svelte'
import { connection } from './connection.svelte'

vi.mock('./changes.svelte', async () => ({
  changes: (await import('./test/feedMock')).feed.changes,
}))

let teardown: (() => void) | null = null

/** Make a resource inside an effect root, as a component's init would */
function mount<T>(
  load: () => Promise<T>,
  options: ResourceOptions<T>,
): Resource<T> {
  let made!: Resource<T>
  teardown = $effect.root(() => {
    made = resource(load, options)
  })
  flushSync()
  return made
}

/** A promise and the hands that settle it */
function deferred<T>() {
  let resolve!: (v: T) => void
  let reject!: (e: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

const settle = () => vi.waitFor(() => {})

function setHidden(hidden: boolean) {
  Object.defineProperty(document, 'visibilityState', {
    configurable: true,
    get: () => (hidden ? 'hidden' : 'visible'),
  })
  document.dispatchEvent(new Event('visibilitychange'))
}

beforeEach(() => {
  workspace.current = 'default'
})
afterEach(() => {
  teardown?.()
  teardown = null
  feed.watches.clear()
  setHidden(false)
  vi.useRealTimers()
})

it('loads once and is ready', async () => {
  const load = vi.fn(() => Promise.resolve(['a']))
  const r = mount(load, { watch: [] })
  expect(r.status).toBe('loading')
  expect(r.data).toBeUndefined()
  await vi.waitFor(() => expect(r.status).toBe('ready'))
  expect(r.data).toEqual(['a'])
  expect(r.stale).toBe(false)
  expect(load).toHaveBeenCalledTimes(1)
})

it('drops a reply overtaken by a workspace switch', async () => {
  const first = deferred<string>()
  const second = deferred<string>()
  const load = vi
    .fn<() => Promise<string>>()
    .mockReturnValueOnce(first.promise)
    .mockReturnValueOnce(second.promise)
  const r = mount(load, { watch: [] })
  workspace.current = 'other'
  flushSync()
  expect(load).toHaveBeenCalledTimes(2)
  second.resolve('other rows')
  await vi.waitFor(() => expect(r.data).toBe('other rows'))
  first.resolve('default rows')
  await settle()
  expect(r.data).toBe('other rows')
  expect(r.status).toBe('ready')
})

it('keeps data across a filter change, clears it across a workspace switch', async () => {
  const filter = $state({ q: 'a' })
  let gate = deferred<string>()
  const load = vi.fn(() => {
    const q = filter.q
    return gate.promise.then((v) => `${q}:${v}`)
  })
  const r = mount(load, { watch: [] })
  gate.resolve('1')
  await vi.waitFor(() => expect(r.data).toBe('a:1'))

  gate = deferred<string>()
  filter.q = 'b'
  flushSync()
  expect(r.status).toBe('loading')
  expect(r.data).toBe('a:1')
  gate.resolve('2')
  await vi.waitFor(() => expect(r.data).toBe('b:2'))

  gate = deferred<string>()
  workspace.current = 'other'
  flushSync()
  expect(r.status).toBe('loading')
  expect(r.data).toBeUndefined()
  gate.resolve('3')
  await vi.waitFor(() => expect(r.data).toBe('b:3'))
})

it('keeps data when a quiet refresh fails', async () => {
  const load = vi
    .fn<() => Promise<string>>()
    .mockResolvedValueOnce('rows')
    .mockRejectedValueOnce(new Error('down'))
    .mockResolvedValueOnce('fresh')
  const r = mount(load, { watch: ['gallery'] })
  await vi.waitFor(() => expect(r.status).toBe('ready'))
  feed.move('default', 'gallery')
  expect(r.status).toBe('ready')
  await vi.waitFor(() => expect(r.stale).toBe(true))
  expect(r.status).toBe('ready')
  expect(r.data).toBe('rows')
  expect(r.error).toBeInstanceOf(Error)
  // The next good read clears it
  feed.move('default', 'gallery')
  await vi.waitFor(() => expect(r.data).toBe('fresh'))
  expect(r.stale).toBe(false)
  expect(r.error).toBeUndefined()
})

it('a quiet failure that overtakes a loud load is an error, not loading', async () => {
  const first = deferred<string>()
  const load = vi
    .fn<() => Promise<string>>()
    .mockReturnValueOnce(first.promise)
    .mockRejectedValueOnce(new Error('down'))
  const r = mount(load, { watch: ['jobs'] })
  feed.move('default', 'jobs')
  await vi.waitFor(() => expect(r.status).toBe('error'))
  first.resolve('late')
  await settle()
  expect(r.data).toBeUndefined()
})

it('a failed first load is an error, and reload is not quiet', async () => {
  const gate = deferred<string>()
  const load = vi
    .fn<() => Promise<string>>()
    .mockRejectedValueOnce(new Error('down'))
    .mockReturnValueOnce(gate.promise)
  const r = mount(load, { watch: [] })
  await vi.waitFor(() => expect(r.status).toBe('error'))
  const again = r.reload()
  expect(r.status).toBe('loading')
  gate.resolve('rows')
  await again
  expect(r.status).toBe('ready')
  expect(r.data).toBe('rows')
})

it('reloads quietly once when the epoch rises', async () => {
  const load = vi.fn(() => Promise.resolve('rows'))
  const r = mount(load, { watch: [] })
  await vi.waitFor(() => expect(r.status).toBe('ready'))
  connection.epoch++
  flushSync()
  expect(r.status).toBe('ready')
  await settle()
  expect(load).toHaveBeenCalledTimes(2)
  flushSync()
  await settle()
  expect(load).toHaveBeenCalledTimes(2)
})

it('reloads quietly on a watched kind and not on another', async () => {
  const load = vi.fn(() => Promise.resolve('rows'))
  const r = mount(load, { watch: ['gallery', 'assets'] })
  await vi.waitFor(() => expect(r.status).toBe('ready'))
  expect(feed.watched('default')).toEqual(['gallery', 'assets'])
  feed.move('default', 'jobs')
  feed.move('other', 'gallery')
  expect(load).toHaveBeenCalledTimes(1)
  feed.move('default', 'assets')
  expect(r.status).toBe('ready')
  expect(load).toHaveBeenCalledTimes(2)
  // A switch moves the watch to the new workspace
  workspace.current = 'other'
  flushSync()
  expect(feed.watched('default')).toEqual([])
  expect(feed.watched('other')).toEqual(['gallery', 'assets'])
})

it('watch: [] holds no feed', () => {
  mount(() => Promise.resolve(1), { watch: [] })
  expect(feed.watches.size).toBe(0)
})

it('live polls at its interval and pauses while hidden', async () => {
  vi.useFakeTimers()
  const load = vi.fn(() => Promise.resolve('rows'))
  const r = mount(load, { live: { everyMs: 1000 } })
  await vi.advanceTimersByTimeAsync(0)
  expect(r.status).toBe('ready')
  expect(load).toHaveBeenCalledTimes(1)
  expect(feed.watches.size).toBe(0)
  await vi.advanceTimersByTimeAsync(1000)
  expect(load).toHaveBeenCalledTimes(2)
  expect(r.status).toBe('ready')
  setHidden(true)
  await vi.advanceTimersByTimeAsync(5000)
  expect(load).toHaveBeenCalledTimes(2)
  // Back: once at once, then on the interval
  setHidden(false)
  expect(load).toHaveBeenCalledTimes(3)
  await vi.advanceTimersByTimeAsync(1000)
  expect(load).toHaveBeenCalledTimes(4)
})

it('alsoLive polls only while it returns a number', async () => {
  vi.useFakeTimers()
  const busy = $state({ on: false })
  const load = vi.fn(() => Promise.resolve('rows'))
  mount(load, {
    watch: ['jobs'],
    alsoLive: () => (busy.on ? 1000 : null),
  })
  await vi.advanceTimersByTimeAsync(3000)
  expect(load).toHaveBeenCalledTimes(1)
  busy.on = true
  flushSync()
  await vi.advanceTimersByTimeAsync(3000)
  expect(load).toHaveBeenCalledTimes(4)
  busy.on = false
  flushSync()
  await vi.advanceTimersByTimeAsync(3000)
  expect(load).toHaveBeenCalledTimes(4)
})

it('merge decides what is held', async () => {
  const load = vi
    .fn<() => Promise<{ rev: number }>>()
    .mockResolvedValueOnce({ rev: 2 })
    .mockResolvedValueOnce({ rev: 1 })
  const merge = vi.fn((held: { rev: number } | undefined, incoming) =>
    held && held.rev > incoming.rev ? held : incoming,
  )
  const r = mount(load, { watch: ['requests'], merge })
  await vi.waitFor(() => expect(r.data).toEqual({ rev: 2 }))
  expect(merge).toHaveBeenCalledWith(undefined, { rev: 2 })
  feed.move('default', 'requests')
  await vi.waitFor(() => expect(merge).toHaveBeenCalledTimes(2))
  expect(r.data).toEqual({ rev: 2 })
})

it('teardown leaves no watcher and no timer', async () => {
  vi.useFakeTimers()
  const load = vi.fn(() => Promise.resolve(1))
  mount(load, { watch: ['jobs'], alsoLive: () => 1000 })
  await vi.advanceTimersByTimeAsync(1000)
  expect(feed.watches.size).toBe(1)
  teardown?.()
  teardown = null
  expect(feed.watches.size).toBe(0)
  const calls = load.mock.calls.length
  await vi.advanceTimersByTimeAsync(10_000)
  expect(load).toHaveBeenCalledTimes(calls)
})

it('watch and live exclude each other, and one is required', () => {
  const f = () => Promise.resolve(1)
  // @ts-expect-error a freshness is required
  expect(() => mount(f, {})).not.toThrow()
  teardown?.()
  // @ts-expect-error live and watch together
  mount(f, { watch: [], live: { everyMs: 1 } })
  teardown?.()
  // @ts-expect-error alsoLive belongs to watch
  mount(f, { live: { everyMs: 1 }, alsoLive: () => 1 })
  teardown?.()
  teardown = null
})

type Row = { id: string; v: number }
const byId = (rows: Row[]) => rows.map((r) => [r.id, r] as const)

it('a quiet reload reports the changed and new keys and clears them', async () => {
  vi.useFakeTimers()
  const load = vi
    .fn<() => Promise<Row[]>>()
    .mockResolvedValueOnce([
      { id: 'a', v: 1 },
      { id: 'b', v: 1 },
    ])
    .mockResolvedValueOnce([
      { id: 'a', v: 1 },
      { id: 'b', v: 2 },
      { id: 'c', v: 1 },
    ])
  const r = mount(load, { watch: ['jobs'], rows: byId })
  await vi.advanceTimersByTimeAsync(0)
  expect(r.status).toBe('ready')
  expect([...r.changed]).toEqual([])
  feed.move('default', 'jobs')
  await vi.advanceTimersByTimeAsync(0)
  expect([...r.changed].sort()).toEqual(['b', 'c'])
  await vi.advanceTimersByTimeAsync(CHANGED_MARK_MS - 1)
  expect(r.changed.size).toBe(2)
  await vi.advanceTimersByTimeAsync(1)
  expect(r.changed.size).toBe(0)
})

it('a non-quiet load marks nothing', async () => {
  const load = vi
    .fn<() => Promise<Row[]>>()
    .mockResolvedValueOnce([{ id: 'a', v: 1 }])
    .mockResolvedValueOnce([{ id: 'a', v: 2 }])
    .mockResolvedValueOnce([{ id: 'a', v: 3 }])
  const r = mount(load, { watch: ['jobs'], rows: byId })
  await vi.waitFor(() => expect(r.status).toBe('ready'))
  expect(r.changed.size).toBe(0)
  feed.move('default', 'jobs')
  await vi.waitFor(() => expect(r.changed.has('a')).toBe(true))
  // A loud reload (Retry) is a fresh look: it takes the marks away
  await r.reload()
  expect(r.data).toEqual([{ id: 'a', v: 3 }])
  expect(r.changed.size).toBe(0)
})

it('a lazy resource does not load until a watched kind moves', async () => {
  const load = vi.fn(() => Promise.resolve('rows'))
  const r = mount(load, { watch: ['prompts'], lazy: true })
  await settle()
  expect(load).not.toHaveBeenCalled()
  expect(r.data).toBeUndefined()
  feed.move('default', 'prompts')
  await vi.waitFor(() => expect(r.data).toBe('rows'))
  expect(load).toHaveBeenCalledTimes(1)
  connection.epoch++
  flushSync()
  await vi.waitFor(() => expect(load).toHaveBeenCalledTimes(2))
})

it("a failed load for a new question drops the old question's data", async () => {
  const filter = $state({ q: 'a' })
  let answer: () => Promise<string> = () => Promise.resolve('rows')
  const load = vi.fn(() => {
    const q = filter.q
    return answer().then((v) => `${q}:${v}`)
  })
  const r = mount(load, { watch: ['jobs'] })
  await vi.waitFor(() => expect(r.data).toBe('a:rows'))

  // The same question failing quietly keeps its rows, stale
  answer = () => Promise.reject(new Error('down'))
  feed.move('default', 'jobs')
  await vi.waitFor(() => expect(r.stale).toBe(true))
  expect(r.data).toBe('a:rows')

  // A new question failing does not leave the old one's rows standing in
  filter.q = 'b'
  flushSync()
  expect(r.data).toBe('a:rows')
  await vi.waitFor(() => expect(r.status).toBe('error'))
  expect(r.data).toBeUndefined()

  // and a quiet run that overtakes that load answers for the new question too
  answer = () => Promise.resolve('rows')
  feed.move('default', 'jobs')
  await vi.waitFor(() => expect(r.data).toBe('b:rows'))
})

it('a lock or an outage on a new question keeps the rows: the banner says it', async () => {
  const { ApiError } = await import('./apiErrors')
  const filter = $state({ q: 'a' })
  let answer: () => Promise<string> = () => Promise.resolve('rows')
  const load = vi.fn(() => {
    const q = filter.q
    return answer().then((v) => `${q}:${v}`)
  })
  const r = mount(load, { watch: [] })
  await vi.waitFor(() => expect(r.data).toBe('a:rows'))
  answer = () =>
    Promise.reject(new ApiError('Invalid or missing API token', 401))
  filter.q = 'b'
  flushSync()
  await vi.waitFor(() => expect(r.status).toBe('error'))
  expect(r.data).toBe('a:rows')
})
