import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import RequestFilter from './RequestFilter.svelte'
import { summary } from '../request/fixtures'
import { feed } from '../test/feedMock'
import { workspace } from '../workspaceState.svelte'

const data = vi.hoisted(() => ({
  requests: [] as unknown[],
  error: null as Error | null,
  asked: [] as string[],
}))
vi.mock('../api', () => ({
  api: {
    listRequests: (ws: string) => {
      data.asked.push(ws)
      return data.error
        ? Promise.reject(data.error)
        : Promise.resolve({ workspace: ws, requests: data.requests })
    },
  },
}))
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))

beforeEach(() => {
  workspace.current = 'studio'
  data.error = null
  data.requests = []
  data.asked = []
  history.replaceState(null, '', '#/ws/studio/gallery')
})
afterEach(cleanup)

const picker = () =>
  screen.queryByRole('combobox', { name: 'filter by request' })

it('offers the requests of the workspace as a picker', async () => {
  data.requests = [summary()]
  render(RequestFilter, { request: '', item: '' })
  await waitFor(() => expect(picker()).toBeTruthy())
  expect(screen.getByRole('option', { name: 'Launch posters' })).toBeTruthy()
})

it('is hidden when the call fails, and says nothing', async () => {
  data.error = new Error('down')
  render(RequestFilter, { request: '', item: '' })
  await new Promise((r) => setTimeout(r, 10))
  expect(picker()).toBeNull()
  expect(screen.queryByRole('alert')).toBeNull()
})

it('a request written from outside appears when the feed says requests moved', async () => {
  render(RequestFilter, { request: '', item: '' })
  await new Promise((r) => setTimeout(r, 10))
  expect(feed.watched('studio')).toEqual(['requests'])
  expect(picker()).toBeNull()
  data.requests = [summary()]
  feed.move('studio', 'requests')
  await waitFor(() => expect(picker()).toBeTruthy())
})

it("with a workspace prop, it lists that workspace's requests", async () => {
  data.requests = [summary()]
  render(RequestFilter, { request: '', item: '', workspace: 'shots' })
  await waitFor(() => expect(picker()).toBeTruthy())
  expect(data.asked).toEqual(['shots'])
  cleanup()
  // and a chosen request links into that workspace
  history.replaceState(null, '', '#/status?req=r1')
  render(RequestFilter, { request: '', item: '', workspace: 'shots' })
  const chip = await screen.findByTestId('req-filter')
  expect(chip.querySelector('a')?.getAttribute('href')).toBe(
    '#/ws/shots/requests/r1',
  )
})

it('offers no picker when it is not pickable', async () => {
  data.requests = [summary()]
  render(RequestFilter, { request: '', item: '', pickable: false })
  await new Promise((r) => setTimeout(r, 10))
  expect(picker()).toBeNull()
})
