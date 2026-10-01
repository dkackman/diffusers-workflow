import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import PromptsPage from './PromptsPage.svelte'
import '../router.svelte'

const detail = (origin: string, writable: boolean) => ({
  description: '',
  intended_model: '',
  tags: [] as string[],
  text: 'a prompt',
  origin,
  writable,
})

const listPrompts = vi.hoisted(() =>
  vi.fn(() =>
    Promise.resolve({
      libraries: [
        { origin: 'workspace', root: '/srv/prompts', writable: true },
        { origin: 'examples', root: '/ex/prompts', writable: false },
      ],
      shadowed: [],
      prompts: ['mine', 'theirs'],
      details: {},
    } as unknown),
  ),
)
vi.mock('../api', () => ({
  api: { listPrompts: () => listPrompts() },
}))

beforeEach(() => {
  location.hash = '#/ws/default/prompts'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  listPrompts.mockResolvedValue({
    libraries: [
      { origin: 'workspace', root: '/srv/prompts', writable: true },
      { origin: 'examples', root: '/ex/prompts', writable: false },
    ],
    shadowed: [],
    prompts: ['mine', 'theirs'],
    details: {
      mine: detail('workspace', true),
      theirs: detail('examples', false),
    },
  })
})
afterEach(cleanup)

it('badges a prompt by the origin on its own detail', async () => {
  render(PromptsPage)
  await waitFor(() => expect(screen.getByText('theirs')).toBeTruthy())
  expect(
    screen.getByTitle('read-only: from the examples library').textContent,
  ).toContain('examples')
  // The workspace's own prompt carries no badge
  expect(screen.queryAllByTitle(/read-only/)).toHaveLength(1)
})

it('names the writable workspace root as where prompts are read from', async () => {
  render(PromptsPage)
  await waitFor(() => expect(screen.getByText('/srv/prompts')).toBeTruthy())
})
