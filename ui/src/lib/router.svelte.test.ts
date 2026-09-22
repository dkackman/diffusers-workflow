import { afterEach, beforeEach, expect, it, vi } from 'vitest'

vi.mock('./api', () => ({
  api: { listWorkspaces: vi.fn(() => new Promise(() => {})) },
}))
vi.mock('./toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

beforeEach(() => {
  localStorage.clear()
  location.hash = ''
})
afterEach(() => vi.resetModules())

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
