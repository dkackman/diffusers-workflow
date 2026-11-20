import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import RequestPage from './RequestPage.svelte'
import { workspace } from '../workspaceState.svelte'
import type { LedgerRequest } from '../types'

const data = vi.hoisted(() => ({
  request: null as unknown,
  getError: null as { status: number; message: string } | null,
  getJob: vi.fn(),
  putRequest: vi.fn(),
}))
vi.mock('../api', () => ({
  ApiError: class ApiError extends Error {
    status: number
    constructor(message: string, status = 500) {
      super(message)
      this.name = 'ApiError'
      this.status = status
    }
  },
  outputUrl: (path: string) => `/outputs/${path}`,
  api: {
    getRequest: vi.fn(async () => {
      if (data.getError) {
        const { ApiError } = await import('../api')
        throw new ApiError(data.getError.message, data.getError.status)
      }
      return data.request
    }),
    getJob: (id: string) => data.getJob(id),
    putRequest: (...args: unknown[]) => data.putRequest(...args),
  },
}))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

const make = (): LedgerRequest =>
  ({
    id: 'r1',
    title: 'Launch posters',
    state: 'open',
    revision: 3,
    prompt: 'make me\n  three posters',
    goal: 'Three finished posters',
    spent_minutes: 5,
    remaining_minutes: 7,
    items: [
      {
        key: 'a',
        title: 'First poster',
        depends_on: [],
        estimate_minutes: 2,
        workflow: 'poster',
        arguments: {},
        dropped: false,
        accepted: null,
        comment: { text: 'nice', at: '2026-01-02T03:04:05Z' },
        status: 'done',
        attempts: [{ job: 'job-a', status: 'succeeded', minutes: 1.5 }],
        result: { job: 'job-a', run_id: 'run-a' },
        spent_minutes: 1.5,
      },
      {
        key: 'b',
        title: 'Second poster',
        depends_on: ['a'],
        estimate_minutes: null,
        workflow: 'poster',
        arguments: {},
        dropped: false,
        accepted: null,
        comment: null,
        status: 'planned',
        attempts: [],
        result: null,
        spent_minutes: 0,
      },
    ],
    history: [
      {
        revision: 1,
        at: '2026-01-01T00:00:00Z',
        by: 'agent',
        why: 'opened',
        goal: 'First goal',
      },
      { revision: 3, at: '2026-01-03T00:00:00Z', by: 'user', why: 'commented' },
    ],
  }) as unknown as LedgerRequest

beforeEach(() => {
  workspace.current = 'studio'
  data.request = make()
  data.getError = null
  data.getJob.mockReset()
  data.getJob.mockResolvedValue({
    id: 'job-a',
    workspace: 'studio',
    manifest: [{ step: 's', files: ['a.png'], role: 'final', reused: false }],
    output_kinds: { 'a.png': 'image' },
  })
  data.putRequest.mockReset()
})
afterEach(cleanup)

it('renders the ask, items in order with badges, attempts, totals and history', async () => {
  const { container } = render(RequestPage, { requestId: 'r1' })
  await screen.findByText('Launch posters')
  expect(screen.getByText(/three posters/).textContent).toBe(
    'make me\n  three posters',
  )
  expect(screen.getByText('Three finished posters')).toBeTruthy()
  const titles = [...container.querySelectorAll('.item .title')].map(
    (e) => e.textContent,
  )
  expect(titles).toEqual(['First poster', 'Second poster'])
  const badges = [...container.querySelectorAll('.item .status')].map(
    (e) => e.textContent,
  )
  expect(badges).toEqual(['done', 'planned'])
  expect(screen.getByText(/est\. ~2 min/)).toBeTruthy()
  expect(screen.getByText(/est\. —/)).toBeTruthy()
  expect(screen.getByRole('link', { name: 'job-a' }).getAttribute('href')).toBe(
    '#/ws/studio/jobs/job-a',
  )
  expect(screen.getByText(/~5 min spent/)).toBeTruthy()
  expect(screen.getByText(/~7 min remaining/)).toBeTruthy()
  const history = [...container.querySelectorAll('.history li')].map(
    (e) => e.querySelector('.num')?.textContent,
  )
  expect(history).toEqual(['r3', 'r1'])
  expect(screen.getByText(/goal: First goal/)).toBeTruthy()
  // the result thumbnail links to the job
  await waitFor(() =>
    expect(container.querySelector('.result img')?.getAttribute('src')).toBe(
      '/outputs/a.png',
    ),
  )
})

it('says a missing request no longer exists', async () => {
  data.getError = { status: 404, message: 'not found' }
  render(RequestPage, { requestId: 'gone' })
  await screen.findByText(/no longer exists/i)
})

async function openComment(label: string) {
  await screen.findByText('Launch posters')
  const buttons = screen.getAllByRole('button', { name: label })
  return buttons[0]
}

it('saves a comment at the read revision, changing only that item', async () => {
  data.putRequest.mockImplementation((_ws, _id, body) =>
    Promise.resolve({ ...make(), revision: body.revision + 1 }),
  )
  render(RequestPage, { requestId: 'r1' })
  await fireEvent.click(await openComment('Add comment'))
  const box = screen.getByLabelText('Comment on b')
  await fireEvent.input(box, { target: { value: 'needs more red' } })
  await fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  await waitFor(() => expect(data.putRequest).toHaveBeenCalled())
  const [ws, id, body] = data.putRequest.mock.calls[0]
  expect(ws).toBe('studio')
  expect(id).toBe('r1')
  expect(body).not.toHaveProperty('prompt')
  expect(body).toMatchObject({
    title: 'Launch posters',
    goal: 'Three finished posters',
    revision: 3,
    why: 'comment on b',
    state: 'open',
  })
  const expected = make().items
  expect(body.items[0]).toEqual(expected[0])
  expect(body.items[1]).toEqual({
    ...expected[1],
    comment: { text: 'needs more red' },
  })
  await screen.findByText('revision 4')
})

it('a stale write shows the message and keeps the draft', async () => {
  const { ApiError } = await import('../api')
  data.putRequest.mockRejectedValue(
    new ApiError(
      'This request changed since you opened it (now revision 5) — reload to see it',
      409,
    ),
  )
  render(RequestPage, { requestId: 'r1' })
  await fireEvent.click(await openComment('Add comment'))
  const box = screen.getByLabelText('Comment on b') as HTMLTextAreaElement
  await fireEvent.input(box, { target: { value: 'my draft' } })
  await fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  await screen.findByText(/now revision 5/)
  expect(screen.getByRole('button', { name: 'Reload' })).toBeTruthy()
  expect(
    (screen.getByLabelText('Comment on b') as HTMLTextAreaElement).value,
  ).toBe('my draft')
})

it('links to its jobs and outputs through the req filter', async () => {
  data.request = make()
  render(RequestPage, { requestId: 'r1' })
  const jobs = await screen.findByText('its jobs')
  expect(jobs.getAttribute('href')).toBe(
    `#/ws/${workspace.current}/jobs?req=r1`,
  )
  expect(screen.getByText('its outputs').getAttribute('href')).toBe(
    `#/ws/${workspace.current}/gallery?req=r1`,
  )
})

it('drops a request that lands after the page moved to another', async () => {
  const { api } = await import('../api')
  let land: (reply: unknown) => void = () => {}
  vi.mocked(api.getRequest).mockImplementationOnce(
    () => new Promise((resolve) => (land = resolve)) as never,
  )
  const view = render(RequestPage, { requestId: 'r1' })
  data.request = { ...make(), id: 'r2', title: 'Second ask' }
  await view.rerender({ requestId: 'r2' })
  await screen.findByText('Second ask')
  land(make())
  await new Promise((r) => setTimeout(r, 10))
  expect(screen.queryByText('Launch posters')).toBeNull()
  expect(screen.getByText('Second ask')).toBeTruthy()
})
