import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import RequestsPage from './RequestsPage.svelte'
import { workspace } from '../workspaceState.svelte'

const data = vi.hoisted(() => ({ requests: [] as unknown[] }))
vi.mock('../api', () => ({
  api: {
    listRequests: () =>
      Promise.resolve({ workspace: 'studio', requests: data.requests }),
  },
}))

beforeEach(() => {
  workspace.current = 'studio'
})
afterEach(cleanup)

it('lists each request with a link, counts and minutes', async () => {
  data.requests = [
    {
      id: 'r1',
      title: 'Launch posters',
      state: 'open',
      revision: 2,
      items: 3,
      status_counts: { done: 2, planned: 1 },
      spent_minutes: 4,
      remaining_minutes: 2,
      updated_at: null,
    },
  ]
  render(RequestsPage)
  const link = await screen.findByRole('link', { name: 'Launch posters' })
  expect(link.getAttribute('href')).toBe('#/ws/studio/requests/r1')
  expect(screen.getByText('r1')).toBeTruthy()
  expect(screen.getByText('open')).toBeTruthy()
  expect(screen.getByText('3 items')).toBeTruthy()
  expect(screen.getByText('2 done · 1 planned')).toBeTruthy()
  expect(screen.getByText(/~4 min spent/)).toBeTruthy()
  expect(screen.getByText(/~2 min remaining/)).toBeTruthy()
})

it('says how requests get opened when there are none', async () => {
  data.requests = []
  render(RequestsPage)
  await waitFor(() => expect(screen.getByText(/set_request/)).toBeTruthy())
})
