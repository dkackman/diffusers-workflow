import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import Sidebar from './Sidebar.svelte'
import { invalidateWorkspaces } from './workspace.svelte'

const listWorkspaces = vi.hoisted(() =>
  vi.fn(() =>
    Promise.resolve({
      workspace_root: '/ws',
      default: 'default',
      workspaces: [
        { name: 'default', default: true, usage: { files: 2, bytes: 2048 } },
        { name: 'studio', default: false },
      ],
    }),
  ),
)
const createWorkspace = vi.hoisted(() => vi.fn())
vi.mock('./api', () => ({
  api: {
    listWorkspaces: () => listWorkspaces(),
    createWorkspace: (n: string) => createWorkspace(n),
  },
}))
vi.mock('./toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

beforeEach(() => {
  localStorage.clear()
  createWorkspace.mockReset()
  // The one router instance the component imported parsed the hash at load;
  // drive it the way the browser does rather than resetting modules, which
  // would hand the component a second Svelte runtime. jsdom fires its own
  // hashchange later, which reparses the same hash
  location.hash = '#/ws/studio/gallery'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  // and the cached listing does not leak between cases
  invalidateWorkspaces()
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

it('a collapsed workspace links to its overview', async () => {
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  await waitFor(() => expect(screen.getByText('default')).toBeTruthy())
  // the link's accessible name carries its usage figure too
  expect(
    screen.getByRole('link', { name: /^default/ }).getAttribute('href'),
  ).toBe('#/ws/default/overview')
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

it('+ new refuses a reserved name without calling the server', async () => {
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  // 'studio' is the route's and renders before the listing lands; 'default'
  // only the listing can supply
  await waitFor(() => expect(screen.getByText('default')).toBeTruthy())
  await fireEvent.click(screen.getByRole('button', { name: /new workspace/i }))
  const input = screen.getByPlaceholderText('name')
  await fireEvent.input(input, { target: { value: 'assets' } })
  await fireEvent.keyDown(input, { key: 'Enter' })
  await waitFor(() => expect(screen.getByText(/reserved/)).toBeTruthy())
  expect(createWorkspace).not.toHaveBeenCalled()
})

it('+ new creates and lands on the new overview', async () => {
  createWorkspace.mockResolvedValue({ name: 'fresh' })
  render(Sidebar, { collapsed: false, onToggle: () => {} })
  // 'studio' is the route's and renders before the listing lands; 'default'
  // only the listing can supply
  await waitFor(() => expect(screen.getByText('default')).toBeTruthy())
  await fireEvent.click(screen.getByRole('button', { name: /new workspace/i }))
  const input = screen.getByPlaceholderText('name')
  await fireEvent.input(input, { target: { value: 'fresh' } })
  await fireEvent.keyDown(input, { key: 'Enter' })
  await waitFor(() => expect(location.hash).toBe('#/ws/fresh/overview'))
})

it('collapsed shows icons only and the toggle reports', async () => {
  const onToggle = vi.fn()
  render(Sidebar, { collapsed: true, onToggle })
  await fireEvent.click(screen.getByRole('button', { name: /expand sidebar/i }))
  expect(onToggle).toHaveBeenCalled()
  expect(screen.queryByText('studio')).toBeNull()
})
