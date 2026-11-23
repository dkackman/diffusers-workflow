import { afterEach, beforeEach, expect, it, vi } from 'vitest'

vi.mock('./api', () => ({
  api: { listWorkspaces: vi.fn(() => new Promise(() => {})) },
}))
vi.mock('./toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

// Each test loads its own router, and an earlier test's would still hear
// every hashchange and stamp the shared history with its own count - so the
// hashchange listeners a test adds are removed after it
const listeners: EventListenerOrEventListenerObject[] = []
const addListener = window.addEventListener.bind(window)
beforeEach(() => {
  localStorage.clear()
  // replaceState moves no hash event into the next test, as `location.hash = ''` does
  history.replaceState(null, '', location.pathname + location.search)
  vi.spyOn(window, 'addEventListener').mockImplementation(
    (type: string, listener: EventListenerOrEventListenerObject, options) => {
      if (type === 'hashchange') listeners.push(listener)
      addListener(type, listener, options)
    },
  )
})
const removers: Array<() => void> = []
afterEach(() => {
  removers.splice(0).forEach((remove) => remove())
  for (const listener of listeners.splice(0))
    window.removeEventListener('hashchange', listener)
  vi.restoreAllMocks()
  vi.resetModules()
})
async function loadRouter() {
  const mod = await import('./router.svelte')
  const add = mod.addLeaveGuard
  return {
    ...mod,
    addLeaveGuard: (guard: () => boolean | Promise<boolean>) => {
      const remove = add(guard)
      removers.push(remove)
      return remove
    },
  }
}

// Waits for the browser's own events, never for a fixed time. jsdom dispatches
// a hashchange from its own timer, so a bare timeout can fire before it. A
// navigation is `nav(change, n)`: run `change`, wait for its `n` hashchange
// events (a declined one is two: the change, then the router's undo), then for
// the router's promise chain, which is all microtasks and so drained before
// the next timer.
const flush = () => new Promise((r) => setTimeout(r, 0))
function nav(change: () => void, events = 1) {
  return new Promise<void>((resolve) => {
    let left = events
    const on = () => {
      if (--left > 0) return
      window.removeEventListener('hashchange', on)
      void flush().then(() => resolve())
    }
    window.addEventListener('hashchange', on)
    change()
  })
}
const go = (hash: string, events = 1) =>
  nav(() => (location.hash = hash), events)

it('redirects a legacy hash to the last-used workspace at load', async () => {
  localStorage.setItem('dw-workspace', 'studio')
  location.hash = '#/gallery'
  const { route } = await import('./router.svelte')
  expect(location.hash).toBe('#/ws/studio/gallery')
  expect(route.view).toEqual({
    kind: 'ws',
    workspace: 'studio',
    section: 'gallery',
    rest: [],
  })
})

it('an empty hash lands on the default overview', async () => {
  const { route } = await import('./router.svelte')
  expect(location.hash).toBe('#/ws/default/overview')
  expect(route.view).toMatchObject({
    kind: 'ws',
    workspace: 'default',
    section: 'overview',
  })
})

it('a hash change updates the view and the current workspace', async () => {
  const { route } = await import('./router.svelte')
  const { workspace } = await import('./workspace.svelte')
  await go('#/ws/studio/jobs/j1')
  expect(route.view).toMatchObject({
    workspace: 'studio',
    section: 'jobs',
    rest: ['j1'],
  })
  expect(workspace.current).toBe('studio')
  await go('#/shared/prompts')
  expect(route.view).toMatchObject({ kind: 'shared', section: 'prompts' })
  expect(workspace.current).toBe('studio')
})

it('goWs builds a link under the current workspace', async () => {
  const { goWs } = await import('./router.svelte')
  await go('#/ws/studio/overview')
  goWs('jobs', 'j1')
  expect(location.hash).toBe('#/ws/studio/jobs/j1')
})

it('a hash query is view state, not part of the route', async () => {
  const { route, hashQuery } = await import('./router.svelte')
  await go('#/ws/studio/jobs?status=failed&q=flux')
  expect(route.view).toMatchObject({ section: 'jobs', rest: [] })
  expect(hashQuery()).toEqual({ status: 'failed', q: 'flux' })
})

it('setHashQuery replaces the query without a navigation', async () => {
  const { route, hashQuery, setHashQuery } = await import('./router.svelte')
  await go('#/ws/studio/jobs')
  const parts = route.parts
  const depth = history.length
  setHashQuery({ status: 'failed', q: '' })
  expect(location.hash).toBe('#/ws/studio/jobs?status=failed')
  setHashQuery({ status: '', q: 'a cat & 50%' })
  expect(hashQuery()).toEqual({ q: 'a cat & 50%' })
  setHashQuery({ q: '' })
  expect(location.hash).toBe('#/ws/studio/jobs')
  await flush()
  expect(history.length).toBe(depth)
  expect(route.parts).toBe(parts)
})

it('asks a guard before a path change and goes on when it allows', async () => {
  const { route, addLeaveGuard } = await loadRouter()
  const guard = vi.fn(() => true)
  addLeaveGuard(guard)
  const depth = history.length
  await go('#/ws/studio/jobs/j1')
  expect(guard).toHaveBeenCalledTimes(1)
  expect(location.hash).toBe('#/ws/studio/jobs/j1')
  expect(route.view).toMatchObject({ section: 'jobs', rest: ['j1'] })
  expect(history.length).toBe(depth + 1)
})

it('restores the hash when a guard declines', async () => {
  const { route, addLeaveGuard } = await loadRouter()
  const ws = await import('./workspace.svelte')
  const apply = vi.spyOn(ws, 'applyRouteWorkspace')
  await flush()
  addLeaveGuard(() => false)
  const before = location.hash
  const parts = route.parts
  apply.mockClear()
  await go('#/ws/studio/jobs/j1', 2)
  expect(location.hash).toBe(before)
  expect(route.parts).toEqual(parts)
  expect(apply).not.toHaveBeenCalled()
})

it('does not guard a query-only change', async () => {
  const { route, addLeaveGuard, setHashQuery } = await loadRouter()
  await go('#/ws/studio/jobs')
  const guard = vi.fn(() => false)
  addLeaveGuard(guard)
  setHashQuery({ q: 'a' })
  await go('#/ws/studio/jobs?q=b')
  expect(guard).not.toHaveBeenCalled()
  expect(location.hash).toBe('#/ws/studio/jobs?q=b')
  expect(route.view).toMatchObject({ section: 'jobs' })
})

it('a removed guard is not asked', async () => {
  const { route, addLeaveGuard } = await loadRouter()
  const guard = vi.fn(() => false)
  const remove = addLeaveGuard(guard)
  remove()
  await go('#/ws/studio/jobs/j1')
  expect(guard).not.toHaveBeenCalled()
  expect(route.view).toMatchObject({ rest: ['j1'] })
})

it('awaits an async guard', async () => {
  const { route, addLeaveGuard } = await loadRouter()
  let answer: (ok: boolean) => void = () => {}
  addLeaveGuard(() => new Promise<boolean>((r) => (answer = r)))
  await go('#/ws/studio/jobs/j1')
  // the page keeps showing the committed route until the answer arrives
  expect(route.view).not.toMatchObject({ rest: ['j1'] })
  answer(true)
  await flush()
  expect(location.hash).toBe('#/ws/studio/jobs/j1')
  expect(route.view).toMatchObject({ rest: ['j1'] })
})

// real history: a declined navigation leaves no duplicate or rewritten entry

it('a declined link click undoes its push and Back/Forward still work', async () => {
  const { route, addLeaveGuard } = await loadRouter()
  await go('#/ws/studio/overview')
  let ok = false
  addLeaveGuard(() => ok)
  const depth = history.length
  await go('#/ws/studio/jobs/j1', 2)
  expect(location.hash).toBe('#/ws/studio/overview')
  expect(route.view).toMatchObject({ section: 'overview' })
  // the pushed entry is a forward entry, not a duplicate of the current one
  expect(history.length).toBe(depth + 1)
  ok = true
  await nav(() => history.forward())
  expect(location.hash).toBe('#/ws/studio/jobs/j1')
  expect(route.view).toMatchObject({ rest: ['j1'] })
  await nav(() => history.back())
  expect(location.hash).toBe('#/ws/studio/overview')
  expect(route.view).toMatchObject({ section: 'overview' })
})

it('a declined link click leaves Back a real step, not a dead press', async () => {
  const { route, addLeaveGuard } = await loadRouter()
  await go('#/ws/studio/overview')
  await go('#/ws/studio/gallery')
  let ok = false
  addLeaveGuard(() => ok)
  await go('#/ws/studio/jobs', 2)
  expect(location.hash).toBe('#/ws/studio/gallery')
  ok = true
  await nav(() => history.back())
  expect(location.hash).toBe('#/ws/studio/overview')
  expect(route.view).toMatchObject({ section: 'overview' })
})

it('a declined Back keeps the older entry and Forward intact', async () => {
  const { route, addLeaveGuard } = await loadRouter()
  await go('#/ws/studio/overview')
  await go('#/ws/studio/gallery')
  const depth = history.length
  let ok = false
  addLeaveGuard(() => ok)
  await nav(() => history.back(), 2)
  expect(location.hash).toBe('#/ws/studio/gallery')
  expect(route.view).toMatchObject({ section: 'gallery' })
  expect(history.length).toBe(depth)
  // the entry behind is still overview: an allowed Back lands on it
  ok = true
  await nav(() => history.back())
  expect(location.hash).toBe('#/ws/studio/overview')
  expect(route.view).toMatchObject({ section: 'overview' })
})

it('a declined Forward is undone and a later allowed Forward goes through', async () => {
  const { route, addLeaveGuard } = await loadRouter()
  await go('#/ws/studio/overview')
  await go('#/ws/studio/gallery')
  await nav(() => history.back())
  expect(location.hash).toBe('#/ws/studio/overview')
  let ok = false
  addLeaveGuard(() => ok)
  await nav(() => history.forward(), 2)
  expect(location.hash).toBe('#/ws/studio/overview')
  expect(route.view).toMatchObject({ section: 'overview' })
  ok = true
  await nav(() => history.forward())
  expect(location.hash).toBe('#/ws/studio/gallery')
  expect(route.view).toMatchObject({ section: 'gallery' })
})

it('keeps the history stamp through setHashQuery', async () => {
  const { setHashQuery } = await loadRouter()
  await go('#/ws/studio/jobs')
  const stamped = history.state
  setHashQuery({ q: 'a' })
  expect(history.state).toEqual(stamped)
  expect(stamped).toHaveProperty('dwIndex')
})

it('renameRoute renames the entry in place and asks the guards for the new path', async () => {
  const { route, renameRoute, addLeaveGuard } = await loadRouter()
  await go('#/ws/studio/edit')
  const depth = history.length
  const stamped = history.state
  const parts = route.parts
  let asked = 0
  addLeaveGuard(() => {
    asked++
    return false
  })

  renameRoute('#/ws/studio/edit/mine')
  await flush()
  expect(location.hash).toBe('#/ws/studio/edit/mine')
  expect(history.length).toBe(depth)
  expect(history.state).toEqual(stamped)
  expect(asked).toBe(0)
  expect(route.view).toMatchObject({ section: 'edit', rest: ['mine'] })
  // the same array, so the shell does not take it as a navigation
  expect(route.parts).toBe(parts)
  expect(route.parts).toEqual(['ws', 'studio', 'edit', 'mine'])

  // the renamed path is the committed one: a link to the bare editor is a
  // change of path, which the guard is asked about and can decline
  await go('#/ws/studio/edit', 2)
  expect(asked).toBe(1)
  expect(location.hash).toBe('#/ws/studio/edit/mine')
})

// a second navigation while a guard is asking, and a declined replace

it('a second navigation while a guard asks is not lost', async () => {
  const { route, addLeaveGuard } = await loadRouter()
  await go('#/ws/studio/overview')
  let answer: (ok: boolean) => void = () => {}
  const remove = addLeaveGuard(() => new Promise<boolean>((r) => (answer = r)))
  await go('#/ws/studio/jobs/a')
  await go('#/ws/studio/jobs/b')
  expect(route.view).toMatchObject({ section: 'overview' })
  answer(true)
  await flush()
  remove()
  expect(route.parts).toEqual(['ws', 'studio', 'jobs', 'b'])
  expect(location.hash).toBe('#/ws/studio/jobs/b')
  // settled on b's own entry: Back from it is a, not overview
  await nav(() => history.back())
  expect(route.parts).toEqual(['ws', 'studio', 'jobs', 'a'])
})

it('declining after two navigations walks back to the committed entry', async () => {
  const { route, addLeaveGuard } = await loadRouter()
  await go('#/ws/studio/overview')
  const parts = [...route.parts]
  let answer: (ok: boolean) => void = () => {}
  addLeaveGuard(() => new Promise<boolean>((r) => (answer = r)))
  await go('#/ws/studio/jobs/a')
  await go('#/ws/studio/jobs/b')
  const walk = vi.spyOn(history, 'go')
  await nav(() => answer(false))
  expect(walk).toHaveBeenCalledTimes(1)
  expect(walk).toHaveBeenCalledWith(-2)
  walk.mockRestore()
  expect(location.hash).toBe('#/ws/studio/overview')
  expect(route.parts).toEqual(parts)
})

it('a declined replaceRoute puts the committed hash back in place', async () => {
  const { route, addLeaveGuard, replaceRoute } = await loadRouter()
  await go('#/ws/studio/overview')
  const stamped = history.state
  const depth = history.length
  addLeaveGuard(() => false)
  const walk = vi.spyOn(history, 'go')
  await nav(() => replaceRoute('#/ws/studio/jobs/j1'))
  expect(walk).not.toHaveBeenCalled()
  walk.mockRestore()
  expect(location.hash).toBe('#/ws/studio/overview')
  expect(history.state).toEqual(stamped)
  expect(history.length).toBe(depth)
  expect(route.view).toMatchObject({ section: 'overview' })
})

it('an allowed replaceRoute replaces the entry without a new one', async () => {
  const { route, addLeaveGuard, replaceRoute } = await loadRouter()
  await go('#/ws/studio/overview')
  const stamped = history.state
  const depth = history.length
  addLeaveGuard(() => true)
  await nav(() => replaceRoute('#/ws/studio/jobs/j1'))
  expect(location.hash).toBe('#/ws/studio/jobs/j1')
  expect(route.view).toMatchObject({ section: 'jobs', rest: ['j1'] })
  expect(history.length).toBe(depth)
  expect(history.state).toEqual(stamped)
})

it('a replaceRoute to the current hash leaves the next push a push', async () => {
  const { addLeaveGuard, replaceRoute } = await loadRouter()
  await go('#/ws/studio/overview')
  replaceRoute('#/ws/studio/overview')
  addLeaveGuard(() => false)
  const walk = vi.spyOn(history, 'go')
  await go('#/ws/studio/jobs/j1', 2)
  expect(walk).toHaveBeenCalledWith(-1)
  expect(location.hash).toBe('#/ws/studio/overview')
})
