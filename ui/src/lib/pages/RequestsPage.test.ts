import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import RequestsPage from './RequestsPage.svelte'
import { workspace } from '../workspaceState.svelte'

const data = vi.hoisted(() => ({
  requests: [] as unknown[],
  requestsFor: null as ((ws: string) => Promise<unknown>) | null,
}))
vi.mock('../api', () => ({
  api: {
    listRequests: (ws: string) =>
      data.requestsFor
        ? data.requestsFor(ws)
        : Promise.resolve({ workspace: 'studio', requests: data.requests }),
  },
}))

beforeEach(() => {
  workspace.current = 'studio'
  data.requestsFor = null
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

it("drops another workspace's list that lands after a switch", async () => {
  const row = (id: string, title: string) => ({
    id,
    title,
    state: 'open',
    revision: 1,
    items: 0,
    status_counts: {},
    spent_minutes: 0,
    remaining_minutes: 0,
    updated_at: null,
  })
  let land: (reply: unknown) => void = () => {}
  data.requestsFor = (ws) =>
    ws === 'studio'
      ? new Promise((resolve) => (land = resolve))
      : Promise.resolve({ workspace: ws, requests: [row('o1', 'Other ask')] })
  render(RequestsPage)
  workspace.current = 'other'
  await screen.findByRole('link', { name: 'Other ask' })
  land({ workspace: 'studio', requests: [row('s1', 'Studio ask')] })
  await new Promise((r) => setTimeout(r, 10))
  expect(screen.queryByText('Studio ask')).toBeNull()
  expect(screen.getByRole('link', { name: 'Other ask' })).toBeTruthy()
})
