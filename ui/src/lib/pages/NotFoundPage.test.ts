import { cleanup, render, screen } from '@testing-library/svelte'
import { tick } from 'svelte'
import { afterEach, expect, it, vi } from 'vitest'
import NotFoundPage from './NotFoundPage.svelte'
import { route } from '../router.svelte'
import { workspace } from '../workspace.svelte'

vi.mock('../api', () => ({
  api: { listWorkspaces: vi.fn(() => new Promise(() => {})) },
}))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

afterEach(cleanup)

it('sets the tab title to Page not found', async () => {
  route.view = { kind: 'missing', path: '/nope' }
  render(NotFoundPage)
  await tick()
  expect(document.title).toBe('Page not found · diffusers-workflow')
})

it('names the path and links the overview', () => {
  workspace.current = 'studio'
  route.view = { kind: 'missing', path: '/ws/studio/nope' }
  render(NotFoundPage)
  expect(
    screen.getByRole('heading', { name: 'Page not found' }),
  ).toBeInTheDocument()
  expect(
    screen.getByText('/ws/studio/nope', { exact: false }).closest('p')
      ?.textContent,
  ).toBe('There is no page at /ws/studio/nope.')
  expect(
    screen.getByRole('link', { name: 'Go to the overview' }),
  ).toHaveAttribute('href', '#/ws/studio/overview')
})
