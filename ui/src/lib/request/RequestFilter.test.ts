import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import RequestFilter from './RequestFilter.svelte'
import { summary } from './fixtures'
import { api } from '../test/apiMock'
import { feed } from '../test/feedMock'
import { workspace } from '../workspaceState.svelte'
import { quiet, useQuietClock } from '../test/quiet'

useQuietClock()

const data = vi.hoisted(() => ({
  requests: [] as unknown[],
  error: null as Error | null,
  asked: [] as string[],
}))
vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
api.listRequests.mockImplementation((ws: string) => {
  data.asked.push(ws)
  return data.error
    ? Promise.reject(data.error)
    : Promise.resolve({ workspace: ws, requests: data.requests })
})
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
  await quiet()
  expect(picker()).toBeNull()
  expect(screen.queryByRole('alert')).toBeNull()
})

it('a request written from outside appears when the feed says requests moved', async () => {
  render(RequestFilter, { request: '', item: '' })
  await quiet()
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

it('keeps the hash and the filter in step, both ways', async () => {
  data.requests = [summary()]
  history.replaceState(null, '', '#/ws/studio/gallery?req=r1/hero')
  render(RequestFilter, { request: '', item: '' })
  const chip = await screen.findByTestId('req-filter')
  history.replaceState(null, '', '#/ws/studio/gallery?req=r2')
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  await waitFor(() => expect(chip.textContent).toMatch(/r2/))
  expect(location.hash).toBe('#/ws/studio/gallery?req=r2')
  await fireEvent.click(
    screen.getByRole('button', { name: 'remove the request filter' }),
  )
  await waitFor(() => expect(location.hash).toBe('#/ws/studio/gallery'))
  expect(screen.queryByTestId('req-filter')).toBeNull()
})

it('offers no picker when it is not pickable', async () => {
  data.requests = [summary()]
  render(RequestFilter, { request: '', item: '', pickable: false })
  await quiet()
  expect(picker()).toBeNull()
})

it("the chip names the request's title", async () => {
  data.requests = [summary()]
  history.replaceState(null, '', '#/ws/studio/gallery?req=r1/hero')
  render(RequestFilter, { request: '', item: '' })
  const chip = await screen.findByTestId('req-filter')
  await waitFor(() =>
    expect(chip.textContent).toMatch(/request: Launch posters/),
  )
  expect(chip.textContent).toMatch(/› hero/)
  expect(chip.querySelector('a')?.getAttribute('title')).toBe(
    'r1 - open this request',
  )
})

it('the id stands in until the title is known', async () => {
  data.error = new Error('down')
  history.replaceState(null, '', '#/ws/studio/gallery?req=r1')
  render(RequestFilter, { request: '', item: '' })
  const chip = await screen.findByTestId('req-filter')
  expect(chip.textContent).toMatch(/request:\s*r1/)
  expect(chip.textContent).not.toMatch(/Launch/)
})
