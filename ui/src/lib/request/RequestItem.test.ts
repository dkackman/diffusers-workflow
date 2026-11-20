import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import RequestItem from './RequestItem.svelte'
import { ApiError, api } from '../api'
import type { LedgerItem } from '../types'

vi.mock('../api', async (original) => {
  const real = await original<typeof import('../api')>()
  return {
    ApiError: real.ApiError,
    outputUrl: (path: string, version?: string, ws?: string) =>
      `/outputs/${path}?v=${version}&ws=${ws}`,
    api: {
      getJob: vi.fn(),
      galleryThumbnailUrl: (name: string, ws?: string) =>
        `/thumb/${name}?ws=${ws}`,
    },
  }
})
const confirmDialog = vi.hoisted(() => vi.fn())
vi.mock('../confirm.svelte', () => ({ confirmDialog }))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

const item = (over: Partial<LedgerItem> = {}): LedgerItem =>
  ({
    key: 'a',
    title: 'Opening shot',
    depends_on: [],
    estimate_minutes: 2,
    workflow: 'shot',
    arguments: {},
    dropped: false,
    accepted: null,
    comment: null,
    status: 'done',
    attempts: [{ job: 'job-a', status: 'succeeded', minutes: 1 }],
    result: { job: 'job-a', run_id: 'run-a' },
    spent_minutes: 1,
    ...over,
  }) as unknown as LedgerItem

const props = (it: LedgerItem) => ({
  item: it,
  workspace: 'studio',
  onsave: vi.fn(),
  onreload: vi.fn(),
})

beforeEach(() => {
  vi.mocked(api.getJob).mockReset()
  vi.mocked(api.getJob).mockResolvedValue({
    id: 'job-a',
    manifest: [],
  } as never)
  confirmDialog.mockReset()
})
afterEach(cleanup)

const settle = () => new Promise((r) => setTimeout(r, 10))

it("shows a video result as the gallery's first-frame thumbnail", async () => {
  vi.mocked(api.getJob).mockResolvedValue({
    id: 'job-a',
    workspace: 'studio',
    manifest: [{ step: 's', files: ['shot/clip.mp4'], role: 'final' }],
    output_kinds: { 'shot/clip.mp4': 'video' },
  } as never)
  const { container } = render(RequestItem, props(item()))
  await waitFor(() =>
    expect(container.querySelector('.result img')?.getAttribute('src')).toBe(
      '/thumb/shot/clip.mp4?ws=studio',
    ),
  )
  expect(container.querySelector('a.result')?.getAttribute('href')).toBe(
    '#/ws/studio/jobs/job-a',
  )
})

it('shows an image result as the image itself', async () => {
  vi.mocked(api.getJob).mockResolvedValue({
    id: 'job-a',
    workspace: 'studio',
    manifest: [{ step: 's', files: ['a.png'], role: 'final' }],
    output_kinds: { 'a.png': 'image' },
  } as never)
  const { container } = render(RequestItem, props(item()))
  await waitFor(() =>
    expect(container.querySelector('.result img')?.getAttribute('src')).toBe(
      '/outputs/a.png?v=job-a&ws=studio',
    ),
  )
})

it('asks for a job once, however often the item is refreshed', async () => {
  vi.mocked(api.getJob).mockResolvedValue({
    id: 'job-a',
    workspace: 'studio',
    manifest: [{ step: 's', files: ['a.wav'], role: 'final' }],
    output_kinds: { 'a.wav': 'audio' },
  } as never)
  const view = render(RequestItem, props(item()))
  await settle()
  for (let i = 0; i < 3; i++) {
    // A poll hands the component a new, equal item
    await view.rerender(props(item()))
    await settle()
  }
  expect(api.getJob).toHaveBeenCalledTimes(1)
  expect(view.container.querySelector('.result')).toBeNull()
})

it("does not show the last result's picture under a new result job", async () => {
  vi.mocked(api.getJob).mockResolvedValueOnce({
    id: 'job-a',
    workspace: 'studio',
    manifest: [{ step: 's', files: ['a.png'], role: 'final' }],
    output_kinds: { 'a.png': 'image' },
  } as never)
  const view = render(RequestItem, props(item()))
  await screen.findByRole('img')
  vi.mocked(api.getJob).mockReturnValueOnce(new Promise(() => {}) as never)
  await view.rerender(
    props(item({ result: { job: 'job-b', run_id: 'run-b' } } as never)),
  )
  await settle()
  expect(screen.queryByRole('img')).toBeNull()
})

it('asks again when the item gets a different result job', async () => {
  vi.mocked(api.getJob).mockResolvedValue({
    id: 'job-a',
    workspace: 'studio',
    manifest: [],
  } as never)
  const view = render(RequestItem, props(item()))
  await settle()
  await view.rerender(
    props(item({ result: { job: 'job-b', run_id: 'run-b' } } as never)),
  )
  await settle()
  expect(vi.mocked(api.getJob).mock.calls.map((c) => c[0])).toEqual([
    'job-a',
    'job-b',
  ])
})

const planned = (over: Partial<LedgerItem> = {}) =>
  item({ status: 'planned', attempts: [], result: null, ...over } as never)
const box = () => screen.getByLabelText('Comment on a') as HTMLTextAreaElement

async function draft(text: string, saved: string | null = null) {
  const given = props(
    planned({
      comment: saved ? { text: saved, at: '2026-01-02T03:04:05Z' } : null,
    } as never),
  )
  render(RequestItem, given)
  await fireEvent.click(
    screen.getByRole('button', {
      name: saved ? 'Edit comment' : 'Add comment',
    }),
  )
  await fireEvent.input(box(), { target: { value: text } })
  return given
}

it('Cancel asks before discarding a draft, and keeps it when declined', async () => {
  await draft('needs more red')
  confirmDialog.mockResolvedValue(false)
  await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  await waitFor(() => expect(confirmDialog).toHaveBeenCalledTimes(1))
  expect(box().value).toBe('needs more red')
  confirmDialog.mockResolvedValue(true)
  await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  await waitFor(() =>
    expect(screen.queryByLabelText('Comment on a')).toBeNull(),
  )
})

it('Cancel does not ask over an empty or unchanged draft', async () => {
  await draft('as saved', 'as saved')
  await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  await waitFor(() =>
    expect(screen.queryByLabelText('Comment on a')).toBeNull(),
  )
  expect(confirmDialog).not.toHaveBeenCalled()
  expect(screen.getByText('as saved')).toBeTruthy()
})

it('offers Reload for a stale revision, keeping the draft', async () => {
  const given = await draft('my draft')
  given.onsave.mockRejectedValue(
    new ApiError('at revision 5, not 3', 409, 'stale_revision'),
  )
  await fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  expect((await screen.findByRole('alert')).textContent).toContain(
    'at revision 5, not 3',
  )
  await fireEvent.click(screen.getByRole('button', { name: 'Reload' }))
  expect(given.onreload).toHaveBeenCalledTimes(1)
  await waitFor(() => expect(screen.queryByRole('alert')).toBeNull())
  expect(box().value).toBe('my draft')
})

it("shows another refusal as the server's message, with no Reload", async () => {
  const given = await draft('my draft')
  given.onsave.mockRejectedValue(
    new ApiError(
      'this workspace has no root folder, so it cannot hold requests',
      409,
      'no_workspace_root',
    ),
  )
  await fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  expect((await screen.findByRole('alert')).textContent).toContain(
    'no root folder',
  )
  expect(screen.queryByRole('button', { name: 'Reload' })).toBeNull()
  expect(box().value).toBe('my draft')
})

it('offers no comment box on a dropped item', () => {
  render(
    RequestItem,
    props(planned({ dropped: true, status: 'dropped' } as never)),
  )
  expect(screen.queryByRole('button', { name: /comment/i })).toBeNull()
})
