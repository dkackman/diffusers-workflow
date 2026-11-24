import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import JobPage from './JobPage.svelte'
import '../router.svelte'
import { attempt, entry, item, request } from '../request/fixtures'
import type { JobDetail, LedgerNote, LedgerRequest } from '../types'

// The job page's note for the agent: a take of a request item gets a box
// about this job, an unlinked job none
const held = vi.hoisted(() => ({
  job: null as JobDetail | null,
  request: null as LedgerRequest | null,
}))
const api = vi.hoisted(() => ({
  getJob: vi.fn(),
  getJobWorkflow: vi.fn(),
  galleryMetadata: vi.fn(),
  outputDownloadUrl: (name: string) => `/download/${name}`,
  eventLog: vi.fn(),
  getRequest: vi.fn(),
  addNote: vi.fn(),
  editNote: vi.fn(),
  deleteNote: vi.fn(),
  notesChannel: vi.fn(),
}))
vi.mock('../api', async () => {
  const { ApiError } = await import('../apiErrors')
  return {
    ApiError,
    TERMINAL_STATUSES: ['succeeded', 'failed', 'cancelled'],
    api,
    outputUrl: (path: string) => `/outputs/${path}`,
    setNote: vi.fn(),
    streamJobEvents: () => () => {},
  }
})
const notify = vi.hoisted(() => ({
  error: vi.fn(),
  success: vi.fn(),
  news: vi.fn(),
}))
vi.mock('../toast', () => ({ notify }))

const at = (minute: number) =>
  `2026-01-01T00:${String(minute).padStart(2, '0')}:00+00:00`
const mine: LedgerNote = {
  id: 'n1',
  text: 'warmer light on this one',
  created: at(5),
  at: at(5),
  refs: [{ item: 's3', job: 'j2', attempt: 2 }],
}
const other: LedgerNote = {
  id: 'n2',
  text: 'the first take was too dark',
  created: at(4),
  at: at(4),
  refs: [{ item: 's3', job: 'j1', attempt: 1 }],
}
const plan = (over: Partial<LedgerRequest> = {}) =>
  request(
    [
      item('s3', {
        title: 'Shot 3',
        status: 'done',
        attempts: [attempt('j1', 'failed'), attempt('j2')],
        notes: [other, mine],
      }),
    ],
    over,
  )

const job = (over: Partial<JobDetail> = {}): JobDetail =>
  ({
    id: 'j2',
    workflow: 'shot',
    status: 'succeeded',
    created_at: 1,
    started_at: 1,
    finished_at: 2,
    workspace: 'studio',
    arguments: {},
    warnings: [],
    manifest: [{ step: 'render', files: ['run/s3.mp4'] }],
    output_kinds: { 'run/s3.mp4': 'video' },
    error: null,
    traceback: null,
    event_count: 0,
    historical: true,
    run_dir: null,
    ...over,
  }) as JobDetail

beforeEach(() => {
  sessionStorage.clear()
  location.hash = '#/ws/studio/jobs/j2'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  for (const call of Object.values(api))
    if (vi.isMockFunction(call)) call.mockReset()
  for (const call of Object.values(notify)) call.mockReset()
  api.getJob.mockImplementation(async () => held.job)
  api.getJobWorkflow.mockRejectedValue(new Error('none'))
  api.galleryMetadata.mockResolvedValue({ metadata: null, inputs: [] })
  api.eventLog.mockResolvedValue({ events: [], truncated: false })
  api.getRequest.mockImplementation(async () => held.request)
  api.notesChannel.mockResolvedValue({ listening: false, last_seen: null })
  api.addNote.mockResolvedValue({ revision: 4 })
  held.request = plan()
})
afterEach(cleanup)

const box = () => screen.getByLabelText('Note for the agent on Shot 3')
async function send(text: string) {
  await fireEvent.input(box(), { target: { value: text } })
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
}

it('shows no note for the agent on a job no request item ran', async () => {
  held.job = job()
  render(JobPage, { jobId: 'j2' })
  await waitFor(() => expect(screen.getByText('Results')).toBeTruthy())
  expect(screen.queryByText('Note for the agent')).toBeNull()
  expect(screen.queryByRole('button', { name: /Note this file/ })).toBeNull()
  expect(api.getRequest).not.toHaveBeenCalled()
})

it("shows only this job's notes, and a new one is about this job", async () => {
  held.job = job({ request: 'r1', item: 's3' })
  render(JobPage, { jobId: 'j2' })
  await waitFor(() =>
    expect(screen.getByText('warmer light on this one')).toBeTruthy(),
  )
  expect(screen.queryByText('the first take was too dark')).toBeNull()
  // One read serves the header's link and the box
  expect(screen.getByRole('link', { name: 'r1 › s3' }).title).toBe(
    'Launch posters',
  )
  expect(api.getRequest).toHaveBeenCalledTimes(1)

  await fireEvent.click(
    screen.getByRole('button', { name: 'Add a note for the agent' }),
  )
  await send('hold the end frame longer')
  await waitFor(() => expect(api.addNote).toHaveBeenCalled())
  expect(api.addNote).toHaveBeenCalledWith('studio', {
    request: 'r1',
    item: 's3',
    text: 'hold the end frame longer',
    refs: [{ item: 's3', job: 'j2' }],
  })
  // and the page reads the request again to show it
  await waitFor(() => expect(api.getRequest).toHaveBeenCalledTimes(2))
})

it('"Note this file" makes the new note about that file of this job', async () => {
  held.job = job({ request: 'r1', item: 's3' })
  render(JobPage, { jobId: 'j2' })
  // the note is one of the viewer's actions on the file
  await fireEvent.click(await screen.findByLabelText('open s3.mp4'))
  await fireEvent.click(
    await screen.findByRole('button', {
      name: 'Note this file (s3.mp4) for the agent',
    }),
  )
  await waitFor(() => expect(box()).toBeTruthy())
  await send('the hand flickers at 0:03')
  await waitFor(() => expect(api.addNote).toHaveBeenCalled())
  expect(api.addNote.mock.calls[0][1].refs).toEqual([
    { item: 's3', job: 'j2', output: 'run/s3.mp4' },
  ])
})

it('offers no "Note this file" while the take is still running', async () => {
  held.job = job({ request: 'r1', item: 's3', status: 'running' })
  render(JobPage, { jobId: 'j2' })
  await screen.findByText('warmer light on this one')
  await fireEvent.click(await screen.findByLabelText('open s3.mp4'))
  await screen.findByRole('dialog', { name: 's3.mp4' })
  expect(screen.queryByRole('button', { name: /Note this file/ })).toBeNull()
})

it('tells when the agent answers a note added here, once', async () => {
  held.job = job({ request: 'r1', item: 's3' })
  held.request = plan({ agent_wrote_at: at(9) })
  render(JobPage, { jobId: 'j2' })
  await screen.findByText('warmer light on this one')
  await fireEvent.click(
    screen.getByRole('button', { name: 'Add a note for the agent' }),
  )
  // The note lands waiting; the agent answers before the next read
  const added: LedgerNote = {
    id: 'n3',
    text: 'a slower push-in',
    created: at(10),
    at: at(10),
    refs: [{ item: 's3', job: 'j2', attempt: 2 }],
  }
  const withNote = (over: Partial<LedgerRequest>) =>
    request(
      [
        item('s3', {
          title: 'Shot 3',
          status: 'done',
          attempts: [attempt('j1', 'failed'), attempt('j2')],
          notes: [other, mine, added],
        }),
      ],
      over,
    )
  held.request = withNote({ revision: 4, agent_wrote_at: at(9) })
  await send('a slower push-in')
  await screen.findByText('a slower push-in')
  expect(notify.news).not.toHaveBeenCalled()

  held.request = withNote({
    revision: 5,
    agent_wrote_at: at(12),
    history: [entry(5, 'agent', 'slowed the push-in')],
  })
  // another write from this page reads the request again
  await fireEvent.click(
    await screen.findByRole('button', { name: 'Add a note for the agent' }),
  )
  await send('thanks')
  await waitFor(() => expect(notify.news).toHaveBeenCalledTimes(1))
  expect(notify.news.mock.calls[0][0]).toBe(
    'The agent updated Launch posters after your note: “slowed the push-in”',
  )
})

it('puts Results straight after the header and the error, the notes under them', async () => {
  held.job = job({
    request: 'r1',
    item: 's3',
    status: 'failed',
    error: 'boom',
    run_dir: 'shot/run-1',
  })
  render(JobPage, { jobId: 'j2' })
  await screen.findByText('warmer light on this one')
  const at = (el: Element | null) => {
    expect(el).toBeTruthy()
    return el!
  }
  const order = [
    at(document.querySelector('h1')),
    at(screen.getByText('The job stopped with an error.')),
    at(screen.getByRole('heading', { name: 'Results' })),
    at(screen.getByText('Your note on this job')),
    at(screen.getByText('warmer light on this one')),
  ]
  for (let i = 1; i < order.length; i++)
    expect(
      order[i - 1].compareDocumentPosition(order[i]) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
})
