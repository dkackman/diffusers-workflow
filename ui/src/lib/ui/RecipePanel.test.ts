import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import { createRawSnippet } from 'svelte'
import RecipePanel from './RecipePanel.svelte'
import { ApiError } from '../apiErrors'
import { api } from '../test/apiMock'
import { ASSET } from '../references'
import type { GalleryMetadata } from '../types'

const reply = vi.hoisted(() => ({
  meta: null as Record<string, unknown> | null,
}))
vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
const galleryMetadata = api.galleryMetadata
const getJob = api.getJob
galleryMetadata.mockImplementation((name: string, workspace?: string) => {
  void name
  void workspace
  return Promise.resolve(reply.meta)
})
getJob.mockImplementation((id: string) =>
  Promise.resolve({
    id,
    status: 'succeeded',
    arguments: { prompt: 'waves at dusk', seed: 7 },
    workflow: 'clip',
    workflow_name: 'video/clip',
  }),
)
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))
vi.mock('../clipboard', () => ({
  copyText: vi.fn(() => Promise.resolve(true)),
}))
vi.mock('../toast', () => ({ notify: { error: vi.fn() } }))

const meta = (fields: Partial<GalleryMetadata>) => ({
  name: 'demo/r1/a.png',
  source: 'output',
  metadata: null,
  job: null,
  run_id: 'r1',
  version: null,
  media: null,
  findings: [],
  linked_jobs: [],
  ...fields,
})

afterEach(() => {
  cleanup()
  galleryMetadata.mockClear()
  getJob.mockClear()
})

const panel = (props: Record<string, unknown> = {}) =>
  render(RecipePanel, {
    props: {
      name: 'demo/r1/a.png',
      source: 'output',
      workspace: 'studio',
      ...props,
    },
  })

it('shows the recipe open, with copy buttons on the prompt and seed', async () => {
  reply.meta = meta({
    metadata: {
      workflow: { id: 'portrait' },
      model_name: 'org/model',
      seed: 42,
      arguments: { prompt: 'a cat', seed: 42, steps: 20 },
    },
    job: { id: 'j1', status: 'succeeded' },
    inputs: [{ kind: 'asset', reference: `${ASSET}faces/hero.png` }],
  })
  panel()
  await screen.findByText('a cat')
  expect(screen.getByRole('heading', { name: 'What made this' })).toBeTruthy()
  expect(document.querySelector('details')).toBeNull()
  expect(screen.getByRole('button', { name: 'Copy the prompt' })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Copy the seed' })).toBeTruthy()
  expect(screen.getByText('steps')).toBeTruthy()
  expect(screen.getByRole('link', { name: 'j1' }).getAttribute('href')).toBe(
    '#/ws/studio/jobs/j1',
  )
  expect(screen.getByText(`${ASSET}faces/hero.png`).tagName).toBe('CODE')
  expect(screen.getByRole('button', { name: /Open as workflow/ })).toBeTruthy()
  expect(screen.getByText('demo/r1/')).toBeTruthy()
  expect(galleryMetadata).toHaveBeenCalledWith('demo/r1/a.png', 'studio')
})

it('a job that left history is not linked', async () => {
  getJob.mockImplementationOnce(() =>
    Promise.reject(new ApiError('Job not found', 404)),
  )
  reply.meta = meta({
    name: 'demo/r1/clip.mp4',
    job: { id: 'j9', status: 'succeeded' },
    media: { kind: 'video', duration_seconds: 75 },
  })
  panel({ name: 'demo/r1/clip.mp4' })
  await waitFor(() => expect(getJob).toHaveBeenCalledWith('j9'))
  await waitFor(() =>
    expect(screen.queryByRole('link', { name: 'j9' })).toBeNull(),
  )
  expect(screen.getByText('j9')).toBeTruthy()
})

it('an embedded image shows no job error', async () => {
  getJob.mockImplementationOnce(() => Promise.reject(new Error('boom')))
  reply.meta = meta({
    metadata: { arguments: { prompt: 'a cat' } },
    job: { id: 'j1', status: 'succeeded' },
  })
  panel()
  await screen.findByText('a cat')
  await waitFor(() => expect(getJob).toHaveBeenCalledWith('j1'))
  // the read has settled: a clip's would say so, an image's needs no job
  await new Promise((r) => setTimeout(r, 0))
  expect(screen.queryByText(/load the job/)).toBeNull()
  expect(screen.getByRole('link', { name: 'j1' })).toBeTruthy()
})

it('with no recipe and no job says so once', async () => {
  reply.meta = meta({})
  panel()
  await screen.findByText('No recipe is saved with this file.')
  expect(
    screen.getAllByText('No recipe is saved with this file.'),
  ).toHaveLength(1)
  expect(
    screen.getByText('A step saves one when its result sets embed_metadata.'),
  ).toBeTruthy()
  expect(getJob).not.toHaveBeenCalled()
})

it('reads the job when one is known, and the Workflow row is its file name', async () => {
  reply.meta = meta({
    metadata: { arguments: { prompt: 'a cat' } },
    job: { id: 'j1', status: 'succeeded' },
  })
  panel()
  await screen.findByText('a cat')
  await waitFor(() => expect(getJob).toHaveBeenCalledWith('j1'))
  expect(await screen.findByText('clip')).toBeTruthy()
  cleanup()
  getJob.mockClear()

  reply.meta = meta({
    name: 'demo/r1/clip.mp4',
    job: { id: 'j9', status: 'succeeded' },
    media: { kind: 'video', duration_seconds: 75 },
  })
  panel({ name: 'demo/r1/clip.mp4' })
  await screen.findByText('waves at dusk')
  expect(getJob).toHaveBeenCalledTimes(1)
  expect(getJob).toHaveBeenCalledWith('j9')
  expect(screen.getByText('clip')).toBeTruthy()
  expect(screen.getByText('1m 15s')).toBeTruthy()
  expect(screen.queryByText('No recipe is saved with this file.')).toBeNull()
})

it('children see the same read', async () => {
  reply.meta = meta({
    source: 'asset',
    name: `${ASSET}hero.png`,
    note: 'the hero',
  })
  const children = createRawSnippet((m: () => GalleryMetadata) => ({
    render: () => `<p class="aside">note: ${m().note}</p>`,
  }))
  panel({ name: 'hero.png', source: 'asset', children })
  await waitFor(() =>
    expect(document.querySelector('.aside')?.textContent).toBe(
      'note: the hero',
    ),
  )
  expect(galleryMetadata).toHaveBeenCalledTimes(1)
  expect(galleryMetadata).toHaveBeenCalledWith(`${ASSET}hero.png`, 'studio')
})

it('children can read the file again: the fresh read, or null when it failed', async () => {
  reply.meta = meta({ note: 'old' })
  let again: () => Promise<GalleryMetadata | null> = async () => null
  const children = createRawSnippet(
    (m: () => GalleryMetadata, reload: () => typeof again) => ({
      render: () => `<p class="aside">note: ${m().note}</p>`,
      setup: () => {
        again = reload()
      },
    }),
  )
  panel({ children })
  await waitFor(() =>
    expect(document.querySelector('.aside')?.textContent).toBe('note: old'),
  )
  reply.meta = meta({ note: 'new' })
  expect((await again())?.note).toBe('new')
  galleryMetadata.mockImplementationOnce(() => Promise.reject(new Error('x')))
  expect(await again()).toBeNull()
})
