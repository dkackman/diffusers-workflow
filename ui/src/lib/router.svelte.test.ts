import { afterEach, beforeEach, expect, it, vi } from 'vitest'

vi.mock('./api', () => ({
  api: { listWorkspaces: vi.fn(() => new Promise(() => {})) },
}))
vi.mock('./toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

beforeEach(() => {
  localStorage.clear()
  location.hash = ''
})
// each test's module keeps its hashchange listener, so its guards are dropped
const removers: Array<() => void> = []
afterEach(() => {
  removers.splice(0).forEach((remove) => remove())
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

const tick = () => new Promise((r) => setTimeout(r, 0))

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
  location.hash = '#/ws/studio/jobs/j1'
  await tick()
  expect(route.view).toMatchObject({
    workspace: 'studio',
    section: 'jobs',
    rest: ['j1'],
  })
  expect(workspace.current).toBe('studio')
  location.hash = '#/shared/prompts'
  await tick()
  expect(route.view).toMatchObject({ kind: 'shared', section: 'prompts' })
  expect(workspace.current).toBe('studio')
})

it('goWs builds a link under the current workspace', async () => {
  const { goWs } = await import('./router.svelte')
  location.hash = '#/ws/studio/overview'
  await tick()
  goWs('jobs', 'j1')
  expect(location.hash).toBe('#/ws/studio/jobs/j1')
})

it('a hash query is view state, not part of the route', async () => {
  const { route, hashQuery } = await import('./router.svelte')
  location.hash = '#/ws/studio/jobs?status=failed&q=flux'
  await tick()
  expect(route.view).toMatchObject({ section: 'jobs', rest: [] })
  expect(hashQuery()).toEqual({ status: 'failed', q: 'flux' })
})

it('setHashQuery replaces the query without a navigation', async () => {
  const { route, hashQuery, setHashQuery } = await import('./router.svelte')
  location.hash = '#/ws/studio/jobs'
  await tick()
  const parts = route.parts
  const depth = history.length
  setHashQuery({ status: 'failed', q: '' })
  expect(location.hash).toBe('#/ws/studio/jobs?status=failed')
  setHashQuery({ status: '', q: 'a cat & 50%' })
  expect(hashQuery()).toEqual({ q: 'a cat & 50%' })
  setHashQuery({ q: '' })
  expect(location.hash).toBe('#/ws/studio/jobs')
  await tick()
  expect(history.length).toBe(depth)
  expect(route.parts).toBe(parts)
})

it('asks a guard before a path change and goes on when it allows', async () => {
  const { route, addLeaveGuard } = await loadRouter()
  const guard = vi.fn(() => true)
  addLeaveGuard(guard)
  const depth = history.length
  location.hash = '#/ws/studio/jobs/j1'
  await tick()
  expect(guard).toHaveBeenCalledTimes(1)
  expect(location.hash).toBe('#/ws/studio/jobs/j1')
  expect(route.view).toMatchObject({ section: 'jobs', rest: ['j1'] })
  expect(history.length).toBe(depth + 1)
})

it('restores the hash when a guard declines', async () => {
  const { route, addLeaveGuard } = await loadRouter()
  const ws = await import('./workspace.svelte')
  const apply = vi.spyOn(ws, 'applyRouteWorkspace')
  await tick()
  addLeaveGuard(() => false)
  const before = location.hash
  const parts = route.parts
  apply.mockClear()
  location.hash = '#/ws/studio/jobs/j1'
  await tick()
  expect(location.hash).toBe(before)
  expect(route.parts).toEqual(parts)
  expect(apply).not.toHaveBeenCalled()
})

it('does not guard a query-only change', async () => {
  const { route, addLeaveGuard, setHashQuery } = await loadRouter()
  location.hash = '#/ws/studio/jobs'
  await tick()
  const guard = vi.fn(() => false)
  addLeaveGuard(guard)
  setHashQuery({ q: 'a' })
  location.hash = '#/ws/studio/jobs?q=b'
  await tick()
  expect(guard).not.toHaveBeenCalled()
  expect(location.hash).toBe('#/ws/studio/jobs?q=b')
  expect(route.view).toMatchObject({ section: 'jobs' })
})

it('a removed guard is not asked', async () => {
  const { route, addLeaveGuard } = await loadRouter()
  const guard = vi.fn(() => false)
  const remove = addLeaveGuard(guard)
  remove()
  location.hash = '#/ws/studio/jobs/j1'
  await tick()
  expect(guard).not.toHaveBeenCalled()
  expect(route.view).toMatchObject({ rest: ['j1'] })
})

it('awaits an async guard', async () => {
  const { route, addLeaveGuard } = await loadRouter()
  let answer: (ok: boolean) => void = () => {}
  addLeaveGuard(() => new Promise<boolean>((r) => (answer = r)))
  location.hash = '#/ws/studio/jobs/j1'
  await tick()
  // the page keeps showing the committed route until the answer arrives
  expect(route.view).not.toMatchObject({ rest: ['j1'] })
  answer(true)
  await tick()
  expect(location.hash).toBe('#/ws/studio/jobs/j1')
  expect(route.view).toMatchObject({ rest: ['j1'] })
})

// real history: a declined navigation leaves no duplicate or rewritten entry
const settleHistory = () => new Promise((r) => setTimeout(r, 20))

it('a declined link click undoes its push and Back/Forward still work', async () => {
  const { route, addLeaveGuard } = await loadRouter()
  location.hash = '#/ws/studio/overview'
  await settleHistory()
  let ok = false
  addLeaveGuard(() => ok)
  const depth = history.length
  location.hash = '#/ws/studio/jobs/j1'
  await settleHistory()
  expect(location.hash).toBe('#/ws/studio/overview')
  expect(route.view).toMatchObject({ section: 'overview' })
  // the pushed entry is a forward entry, not a duplicate of the current one
  expect(history.length).toBe(depth + 1)
  ok = true
  history.forward()
  await settleHistory()
  expect(location.hash).toBe('#/ws/studio/jobs/j1')
  expect(route.view).toMatchObject({ rest: ['j1'] })
  history.back()
  await settleHistory()
  expect(location.hash).toBe('#/ws/studio/overview')
  expect(route.view).toMatchObject({ section: 'overview' })
})

it('a declined link click leaves Back a real step, not a dead press', async () => {
  const { route, addLeaveGuard } = await loadRouter()
  location.hash = '#/ws/studio/overview'
  await settleHistory()
  location.hash = '#/ws/studio/gallery'
  await settleHistory()
  let ok = false
  addLeaveGuard(() => ok)
  location.hash = '#/ws/studio/jobs'
  await settleHistory()
  expect(location.hash).toBe('#/ws/studio/gallery')
  ok = true
  history.back()
  await settleHistory()
  expect(location.hash).toBe('#/ws/studio/overview')
  expect(route.view).toMatchObject({ section: 'overview' })
})

it('a declined Back keeps the older entry and Forward intact', async () => {
  const { route, addLeaveGuard } = await loadRouter()
  location.hash = '#/ws/studio/overview'
  await settleHistory()
  location.hash = '#/ws/studio/gallery'
  await settleHistory()
  const depth = history.length
  let ok = false
  addLeaveGuard(() => ok)
  history.back()
  await settleHistory()
  expect(location.hash).toBe('#/ws/studio/gallery')
  expect(route.view).toMatchObject({ section: 'gallery' })
  expect(history.length).toBe(depth)
  // the entry behind is still overview: an allowed Back lands on it
  ok = true
  history.back()
  await settleHistory()
  expect(location.hash).toBe('#/ws/studio/overview')
  expect(route.view).toMatchObject({ section: 'overview' })
})

it('a declined Forward is undone and a later allowed Forward goes through', async () => {
  const { route, addLeaveGuard } = await loadRouter()
  location.hash = '#/ws/studio/overview'
  await settleHistory()
  location.hash = '#/ws/studio/gallery'
  await settleHistory()
  history.back()
  await settleHistory()
  expect(location.hash).toBe('#/ws/studio/overview')
  let ok = false
  addLeaveGuard(() => ok)
  history.forward()
  await settleHistory()
  expect(location.hash).toBe('#/ws/studio/overview')
  expect(route.view).toMatchObject({ section: 'overview' })
  ok = true
  history.forward()
  await settleHistory()
  expect(location.hash).toBe('#/ws/studio/gallery')
  expect(route.view).toMatchObject({ section: 'gallery' })
})

it('keeps the history stamp through setHashQuery', async () => {
  const { setHashQuery } = await loadRouter()
  location.hash = '#/ws/studio/jobs'
  await settleHistory()
  const stamped = history.state
  setHashQuery({ q: 'a' })
  expect(history.state).toEqual(stamped)
  expect(stamped).toHaveProperty('dwIndex')
})
