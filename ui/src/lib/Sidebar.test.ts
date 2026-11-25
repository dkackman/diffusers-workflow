import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import Sidebar from './Sidebar.svelte'
import { setHashQuery } from './router.svelte'
import { invalidateWorkspaces, workspace } from './workspace.svelte'
import { api } from './test/apiMock'
import { feed } from './test/feedMock'
import { CLOCK_TICK_MS } from './clock.svelte'
import { quiet, useQuietClock } from './test/quiet'

useQuietClock()

vi.mock('./api', async () => (await import('./test/apiMock')).apiModule())
api.listWorkspaces.mockImplementation(() =>
  Promise.resolve({
    workspace_root: '/ws',
    default: 'default',
    workspaces: [
      { name: 'default', default: true, usage: { files: 2, bytes: 2048 } },
      { name: 'studio', default: false },
    ],
  }),
)
vi.mock('./changes.svelte', async () => ({
  changes: (await import('./test/feedMock')).feed.changes,
}))
vi.mock('./toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

beforeEach(() => {
  localStorage.clear()
  api.createWorkspace.mockReset()
  // The one router instance the component imported parsed the hash at load;
  // drive it the way the browser does rather than resetting modules, which
  // would hand the component a second Svelte runtime. jsdom fires its own
  // hashchange later, which reparses the same hash
  location.hash = '#/ws/studio/gallery'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  // The listing is module state: drop the cached promise and the fields it
  // filled, so every case really waits on the mocked listing rather than
  // reading what the last one left
  invalidateWorkspaces()
  workspace.names = undefined
  workspace.root = null
  workspace.listed = false
  workspace.usage = {}
})
afterEach(cleanup)

it('lists workspaces, expands the current one and marks its section', async () => {
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  // 'studio' is the route's and renders before the listing lands; 'default'
  // only the listing can supply
  await waitFor(() => expect(screen.getByText('default')).toBeTruthy())
  const gallery = screen.getByRole('link', { name: /gallery/i })
  expect(gallery.getAttribute('href')).toBe('#/ws/studio/gallery')
  expect(gallery.getAttribute('aria-current')).toBe('page')
  // the other workspace is collapsed: its sections are not rendered
  expect(screen.getAllByRole('link', { name: /workflows/i })).toHaveLength(1)
  // usage shows beside a workspace the server measured
  expect(screen.getByText('2 KB')).toBeTruthy()
})

it('a shut workspace links to the same section', async () => {
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  await waitFor(() => expect(screen.getByText('default')).toBeTruthy())
  // the route is studio's gallery, so default's row opens default's gallery;
  // the link's accessible name carries its usage figure too
  expect(
    screen.getByRole('link', { name: /^default/ }).getAttribute('href'),
  ).toBe('#/ws/default/gallery')
})

it('shared and server groups link to their sections', async () => {
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  expect(
    screen.getByRole('link', { name: /prompts/i }).getAttribute('href'),
  ).toBe('#/shared/prompts')
  expect(
    screen.getByRole('link', { name: /examples/i }).getAttribute('href'),
  ).toBe('#/shared/examples')
  expect(
    screen.getByRole('link', { name: /models/i }).getAttribute('href'),
  ).toBe('#/server/models')
  expect(
    screen.getByRole('link', { name: /status/i }).getAttribute('href'),
  ).toBe('#/server/status')
})

it('new opens the form in its place, and closing it brings new back empty', async () => {
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  await waitFor(() => expect(screen.getByText('default')).toBeTruthy())
  await fireEvent.click(screen.getByRole('button', { name: 'new workspace' }))
  expect(screen.queryByRole('button', { name: 'new workspace' })).toBeNull()
  const input = screen.getByPlaceholderText('name')
  await fireEvent.input(input, { target: { value: 'assets' } })
  await fireEvent.keyDown(input, { key: 'Enter' })
  await waitFor(() => expect(screen.getByText(/reserved/)).toBeTruthy())
  await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  expect(screen.queryByPlaceholderText('name')).toBeNull()
  // a reopened form starts over: no draft, no old refusal
  await fireEvent.click(screen.getByRole('button', { name: 'new workspace' }))
  expect((screen.getByPlaceholderText('name') as HTMLInputElement).value).toBe(
    '',
  )
  expect(screen.queryByText(/reserved/)).toBeNull()
})

it('+ new creates and lands on the new overview', async () => {
  api.createWorkspace.mockResolvedValue({ name: 'fresh' })
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  await waitFor(() => expect(screen.getByText('default')).toBeTruthy())
  await fireEvent.click(screen.getByRole('button', { name: /new workspace/i }))
  const input = screen.getByPlaceholderText('name')
  await fireEvent.input(input, { target: { value: 'fresh' } })
  await fireEvent.keyDown(input, { key: 'Enter' })
  await waitFor(() => expect(location.hash).toBe('#/ws/fresh/overview'))
  // the form closed: new is back
  await waitFor(() =>
    expect(screen.getByRole('button', { name: 'new workspace' })).toBeTruthy(),
  )
})

it('no workspace root says why there is no new', async () => {
  api.listWorkspaces.mockResolvedValueOnce({
    workspace_root: null as unknown as string,
    default: 'default',
    workspaces: [{ name: 'default', default: true }],
  })
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  const why = await screen.findByText('New workspaces need a workspace folder')
  expect(why.getAttribute('title')).toBe(
    'Start the server with --workspace <folder> to add workspaces.',
  )
  expect(screen.queryByRole('button', { name: 'new workspace' })).toBeNull()
})

it('a failed listing does not claim there is no workspace folder', async () => {
  api.listWorkspaces.mockRejectedValueOnce(new Error('down'))
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  await waitFor(() => expect(screen.getByText('default')).toBeTruthy())
  expect(workspace.listed).toBe(false)
  expect(
    screen.queryByText('New workspaces need a workspace folder'),
  ).toBeNull()
})

it('collapsing the sidebar keeps a typed workspace name', async () => {
  const view = render(Sidebar, { collapsed: false, onToggle: () => {} })
  await fireEvent.click(
    await screen.findByRole('button', { name: 'new workspace' }),
  )
  await fireEvent.input(screen.getByPlaceholderText('name'), {
    target: { value: 'half-typed' },
  })
  await view.rerender({ collapsed: true, onToggle: () => {} })
  expect(screen.queryByPlaceholderText('name')).toBeNull()
  await view.rerender({ collapsed: false, onToggle: () => {} })
  expect((screen.getByPlaceholderText('name') as HTMLInputElement).value).toBe(
    'half-typed',
  )
})

it('a server with a workspace root offers new, not the reason', async () => {
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  await screen.findByRole('button', { name: 'new workspace' })
  expect(
    screen.queryByText('New workspaces need a workspace folder'),
  ).toBeNull()
})

it('collapsed, every workspace can be chosen', async () => {
  render(Sidebar, { collapsed: true, onToggle: () => {} })
  const dot = await screen.findByRole('link', { name: 'default workspace' })
  expect(dot.getAttribute('title')).toBe('default')
  // the switch keeps the section here too
  expect(dot.getAttribute('href')).toBe('#/ws/default/gallery')
  expect(dot.querySelector('.dot.solo')).not.toBeNull()
  // the open workspace's dot is drawn, not linked: its sections are the links
  expect(screen.queryByRole('link', { name: 'studio workspace' })).toBeNull()
})

it('collapsed shows icons only and the toggle reports', async () => {
  const onToggle = vi.fn()
  render(Sidebar, { collapsed: true, onToggle })
  await fireEvent.click(screen.getByRole('button', { name: /expand sidebar/i }))
  expect(onToggle).toHaveBeenCalled()
  expect(screen.queryByText('studio')).toBeNull()
})

it('the first render is not a switch; changing workspace marks the new one', async () => {
  const { container } = render(Sidebar, {
    collapsed: false,
    onToggle: () => {},
  })
  await waitFor(() => expect(screen.getByText('default')).toBeTruthy())
  expect(container.querySelector('.ws.open.flash')).toBeNull()
  location.hash = '#/ws/default/overview'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  await waitFor(() =>
    expect(
      container.querySelector('.ws.open.flash .wsname')?.textContent,
    ).toContain('default'),
  )
  // the sections of the newly expanded workspace are the ones left once
  // the old block's outro has run
  await waitFor(() =>
    expect(
      screen.getByRole('link', { name: /gallery/i }).getAttribute('href'),
    ).toBe('#/ws/default/gallery'),
  )
})

it('marks the open workspace apart from its siblings', async () => {
  const { container } = render(Sidebar, {
    collapsed: false,
    onToggle: () => {},
  })
  await waitFor(() => expect(screen.getByText('default')).toBeTruthy())
  expect(screen.getByText('Workspaces')).toBeTruthy()
  // one open block holding the name and the sections, the rest shut links
  const open = container.querySelector('.ws.open')!
  expect(open.querySelector('.name')?.textContent).toBe('studio')
  expect(open.querySelector('.sections')).not.toBeNull()
  expect(container.querySelector('.ws.shut .name')?.textContent).toBe('default')
  // each carries its own identity hue
  const hue = (el: Element | null) =>
    (el as HTMLElement | null)?.style.getPropertyValue('--ws-dot')
  expect(hue(open)).toMatch(/^oklch/)
  expect(hue(container.querySelector('.ws.shut'))).not.toBe(hue(open))
})

it('folds a run of same-prefix workspaces into one group row', async () => {
  api.listWorkspaces.mockResolvedValueOnce({
    workspace_root: '/ws',
    default: 'main',
    workspaces: ['regression-a', 'regression-b', 'regression-c', 'main'].map(
      (name) => ({ name, default: name === 'main' }),
    ),
  })
  location.hash = '#/ws/main/overview'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  const group = await screen.findByRole('button', { name: /regression/ })
  expect(group.getAttribute('aria-expanded')).toBe('false')
  expect(screen.queryByText('regression-a')).toBeNull()
  await fireEvent.click(group)
  expect(screen.getByText('regression-a')).toBeTruthy()
})

it('a workspace made or deleted elsewhere shows when the workspaces counter moves', async () => {
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  await waitFor(() => expect(screen.getByText('default')).toBeTruthy())
  expect(feed.watched(workspace.current)).toContain('workspaces')
  const calls = api.listWorkspaces.mock.calls.length
  api.listWorkspaces.mockImplementationOnce(() =>
    Promise.resolve({
      workspace_root: '/ws',
      default: 'default',
      workspaces: [
        { name: 'default', default: true, usage: { files: 2, bytes: 2048 } },
        { name: 'studio', default: false },
        { name: 'agent-made', default: false },
      ],
    }),
  )
  feed.move(workspace.current, 'workspaces')
  // the cached listing is dropped and read again, once
  await waitFor(() => expect(screen.getByText('agent-made')).toBeTruthy())
  expect(api.listWorkspaces.mock.calls.length).toBe(calls + 1)
  // another kind moving reads nothing
  feed.move(workspace.current, 'requests')
  expect(api.listWorkspaces.mock.calls.length).toBe(calls + 1)
})

it('a workspace made elsewhere appears', async () => {
  // Another view read the listing first: the cache holds it
  const { loadWorkspaces } = await import('./workspace.svelte')
  await loadWorkspaces()
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  await waitFor(() => expect(screen.getByText('default')).toBeTruthy())
  api.listWorkspaces.mockImplementationOnce(() =>
    Promise.resolve({
      workspace_root: '/ws',
      default: 'default',
      workspaces: [
        { name: 'default', default: true },
        { name: 'studio', default: false },
        { name: 'agent-made', default: false },
      ],
    }),
  )
  // the move drops that cached listing, so the sidebar reads the server's
  feed.move(workspace.current, 'workspaces')
  await waitFor(() => expect(screen.getByText('agent-made')).toBeTruthy())
})

it('Sidebar says when each workspace changed', async () => {
  const now = Math.floor(Date.now() / 1000)
  api.listWorkspaces.mockResolvedValueOnce({
    workspace_root: '/ws',
    default: 'default',
    workspaces: [
      { name: 'default', default: true, mtime: now - 3 * 86_400 } as never,
      { name: 'studio', default: false, mtime: now - 5 * 60 } as never,
    ],
  })
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  await waitFor(() => expect(screen.getByText('default')).toBeTruthy())
  const stampOf = (name: string) =>
    screen
      .getByText(name, { selector: '.name' })
      .parentElement!.querySelector('time')
  // the open row and a shut one, each relative with the exact time on hover
  await waitFor(() =>
    expect(stampOf('studio')?.textContent?.trim()).toBe('changed 5 min ago'),
  )
  expect(stampOf('default')?.textContent?.trim()).toBe('changed 3 days ago')
  expect(stampOf('default')?.getAttribute('title')).toBeTruthy()
})

it('a change in the open workspace moves its Sidebar time without a page reload', async () => {
  // Real time still runs (waitFor), but the clock's tick is the test's
  vi.useFakeTimers({ shouldAdvanceTime: true })
  try {
    const now = Math.floor(Date.now() / 1000)
    const listed = (studio: number) => ({
      workspace_root: '/ws',
      default: 'default',
      workspaces: [
        { name: 'default', default: true },
        { name: 'studio', default: false, mtime: studio } as never,
      ],
    })
    api.listWorkspaces.mockResolvedValueOnce(listed(now - 3 * 3600))
    render(Sidebar, { collapsed: false, onToggle: () => {} })
    const stamp = () =>
      screen
        .getByText('studio', { selector: '.name' })
        .parentElement!.querySelector('time')
        ?.textContent?.trim()
    await waitFor(() => expect(stamp()).toBe('changed 3 h ago'))
    const calls = api.listWorkspaces.mock.calls.length
    // An output lands in the open workspace: read again, quietly, at once
    api.listWorkspaces.mockResolvedValueOnce(listed(now))
    feed.move(workspace.current, 'gallery')
    await waitFor(() => expect(stamp()).toBe('changed just now'))
    expect(api.listWorkspaces.mock.calls.length).toBe(calls + 1)
    // A second move in the same clock tick waits for the next one
    api.listWorkspaces.mockResolvedValueOnce(listed(now))
    feed.move(workspace.current, 'workflows')
    await vi.advanceTimersByTimeAsync(0)
    expect(api.listWorkspaces.mock.calls.length).toBe(calls + 1)
    await vi.advanceTimersByTimeAsync(CLOCK_TICK_MS)
    expect(api.listWorkspaces.mock.calls.length).toBe(calls + 2)
    // and a tick with nothing owed reads nothing
    await vi.advanceTimersByTimeAsync(CLOCK_TICK_MS)
    expect(api.listWorkspaces.mock.calls.length).toBe(calls + 2)
  } finally {
    vi.useRealTimers()
  }
})

it('one feed poll that moves the workspaces and a content kind lists them once', async () => {
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  await waitFor(() => expect(screen.getByText('default')).toBeTruthy())
  const calls = api.listWorkspaces.mock.calls.length
  // a copy into the open workspace: one poll, both counters moved, each
  // watcher called in turn (changes.svelte.ts)
  feed.move(workspace.current, 'workspaces')
  feed.move(workspace.current, 'assets')
  await quiet()
  // the content resource joins the relist on its way
  expect(api.listWorkspaces.mock.calls.length).toBe(calls + 1)
})

it('a content move does not join a listing read started before it', async () => {
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  await waitFor(() => expect(screen.getByText('default')).toBeTruthy())
  const calls = api.listWorkspaces.mock.calls.length
  // the workspaces read leaves, and is still on its way when assets move
  const original = api.listWorkspaces.getMockImplementation()!
  let land = () => {}
  api.listWorkspaces.mockImplementationOnce(
    () => new Promise((resolve) => (land = () => resolve(original()))),
  )
  feed.move(workspace.current, 'workspaces')
  await quiet()
  feed.move(workspace.current, 'assets')
  await quiet()
  expect(api.listWorkspaces.mock.calls.length).toBe(calls + 2)
  land()
})

it("the open workspace's time follows the route's feed", async () => {
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  await waitFor(() => expect(screen.getByText('default')).toBeTruthy())
  // `current` moved off the route's workspace (studio)
  workspace.current = 'default'
  await quiet()
  const calls = api.listWorkspaces.mock.calls.length
  feed.move('studio', 'assets')
  await waitFor(() =>
    expect(api.listWorkspaces.mock.calls.length).toBe(calls + 1),
  )
})

it('a folded group while collapsed shows each of its workspaces', async () => {
  api.listWorkspaces.mockResolvedValueOnce({
    workspace_root: '/ws',
    default: 'main',
    workspaces: ['regression-a', 'regression-b', 'regression-c', 'main'].map(
      (name) => ({ name, default: name === 'main' }),
    ),
  })
  location.hash = '#/ws/main/overview'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  render(Sidebar, { collapsed: true, onToggle: () => {} })
  for (const name of ['regression-a', 'regression-b', 'regression-c'])
    expect(
      await screen.findByRole('link', { name: `${name} workspace` }),
    ).toBeTruthy()
  expect(screen.queryByRole('button', { name: /regression/ })).toBeNull()
})

it('re-expand returns focus to the collapse button', async () => {
  const onToggle = vi.fn()
  const { rerender } = render(Sidebar, { collapsed: true, onToggle })
  // jsdom, like Safari, does not focus a button on click
  await fireEvent.click(screen.getByRole('button', { name: 'expand sidebar' }))
  await rerender({ collapsed: false, onToggle })
  expect(onToggle).toHaveBeenCalledOnce()
  expect(document.activeElement).toBe(
    screen.getByRole('button', { name: 'collapse sidebar' }),
  )
})

it("a shut workspace's link follows a filter typed after it was drawn", async () => {
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  const shut = await screen.findByRole('link', { name: /default/ })
  setHashQuery({ q: 'cat', open: 'a.png' })
  await waitFor(() =>
    expect(shut.getAttribute('href')).toBe('#/ws/default/gallery?q=cat'),
  )
})
