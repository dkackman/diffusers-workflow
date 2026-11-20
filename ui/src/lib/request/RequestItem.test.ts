import { cleanup, render, waitFor } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import RequestItem from './RequestItem.svelte'
import { api } from '../api'
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

beforeEach(() => vi.mocked(api.getJob).mockReset())
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
