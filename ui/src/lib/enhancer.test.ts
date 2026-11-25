import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { ApiError } from './apiErrors'
import { reportResponse, reportUnreachable } from './connection.svelte'
import { Enhancer } from './enhancer.svelte'
import { api } from './test/apiMock'
import type { EnhancerPreset } from './types'

vi.mock('./api', async () =>
  (await import('./test/apiMock')).apiModule({ fetchOutputText: vi.fn() }),
)
api.listModels.mockResolvedValue({ repos: [] })
api.startDownload.mockResolvedValue({})

const preset = (key: string, intended: string[] = []): EnhancerPreset => ({
  key,
  label: key,
  default_model: `org/${key}`,
  models: [`org/${key}`],
  intended_models: intended,
  placeholder: '',
})

beforeEach(() => {
  api.listEnhancers.mockResolvedValue({
    presets: [preset('h3', ['minimax-h3']), preset('ltx', ['ltx-2'])],
  })
})
afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

// streamJobEvents is the real one; only the browser's EventSource stands in
class FakeSource {
  static all: FakeSource[] = []
  onmessage: ((message: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  closed = false
  constructor(readonly url: string) {
    FakeSource.all.push(this)
  }
  close() {
    this.closed = true
  }
  send(event: object) {
    this.onmessage?.({ data: JSON.stringify(event) })
  }
}

const queued = { id: 'j1', workspace: 'default', status: 'queued' }

async function generating() {
  FakeSource.all = []
  vi.stubGlobal('EventSource', FakeSource)
  api.enhance.mockResolvedValue(queued)
  const enhancer = new Enhancer()
  enhancer.idea = 'a cat'
  await enhancer.generate()
  return enhancer
}
const settle = () => new Promise((resolve) => setTimeout(resolve))

it('ends when a dropped stream finds the job gone', async () => {
  const enhancer = await generating()
  api.getJob.mockRejectedValue(new ApiError('Unknown job', 404))
  FakeSource.all[0].onerror?.()
  await settle()
  expect(enhancer.busy).toBe(false)
  expect(enhancer.status).toBe('')
  expect(enhancer.jobId).toBe('')
  expect(enhancer.error).toBe(
    "Couldn't read the enhancement: The job wasn't found",
  )
  expect(FakeSource.all[0].closed).toBe(true)
})

it('ends when a dropped stream finds the job finished', async () => {
  const enhancer = await generating()
  api.getJob.mockResolvedValue({ ...queued, status: 'failed', error: 'oom' })
  FakeSource.all[0].onerror?.()
  await settle()
  expect(enhancer.busy).toBe(false)
  expect(enhancer.error).toBe('The enhancement failed')
  expect(enhancer.errorDetail).toBe('oom')
  expect(FakeSource.all[0].closed).toBe(true)
})

it('keeps following through a drop of a job still running', async () => {
  const enhancer = await generating()
  api.getJob.mockResolvedValue({ ...queued, status: 'running' })
  FakeSource.all[0].onerror?.()
  await settle()
  api.getJob.mockRejectedValue(new TypeError('Failed to fetch'))
  FakeSource.all[0].onerror?.()
  await settle()
  expect(enhancer.busy).toBe(true)
  expect(enhancer.error).toBe('')
  expect(FakeSource.all[0].closed).toBe(false)
  enhancer.stop()
})

it('reopens from the last event on a reconnect, until stopped', async () => {
  const enhancer = await generating()
  FakeSource.all[0].send({ seq: 4, event: 'log', message: 'loading' })
  reportUnreachable()
  reportResponse(200)
  expect(FakeSource.all).toHaveLength(2)
  expect(FakeSource.all[0].closed).toBe(true)
  expect(FakeSource.all[1].url).toContain('after=4')
  enhancer.stop()
  expect(FakeSource.all[1].closed).toBe(true)
  reportUnreachable()
  reportResponse(200)
  expect(FakeSource.all).toHaveLength(2)
})

it('opens no stream when stopped while the job is being queued', async () => {
  FakeSource.all = []
  vi.stubGlobal('EventSource', FakeSource)
  let answer!: (job: object) => void
  api.enhance.mockReturnValue(new Promise((resolve) => (answer = resolve)))
  const enhancer = new Enhancer()
  enhancer.idea = 'a cat'
  const done = enhancer.generate()
  enhancer.stop()
  answer(queued)
  await done
  expect(FakeSource.all).toHaveLength(0)
  expect(enhancer.busy).toBe(false)
})

it('loads the presets, and says so when the server has none', async () => {
  const enhancer = new Enhancer()
  await enhancer.load()
  expect(enhancer.presets.map((p) => p.key)).toEqual(['h3', 'ltx'])
  expect(enhancer.down).toBe(false)

  api.listEnhancers.mockResolvedValue({ presets: [] })
  const empty = new Enhancer()
  await empty.load()
  expect(empty.down).toBe(true)
})

it('preselects the preset an intended model names, else keeps the pick', async () => {
  const enhancer = new Enhancer()
  await enhancer.load()
  enhancer.preselect(undefined)
  expect(enhancer.presetKey).toBe('h3')
  expect(enhancer.model).toBe('org/h3')
  enhancer.preselect('ltx-2')
  expect(enhancer.presetKey).toBe('ltx')
  enhancer.preselect('unknown-model')
  expect(enhancer.presetKey).toBe('ltx')
})

it('leaves a hand-typed model alone when the same preset is picked again', async () => {
  const enhancer = new Enhancer()
  await enhancer.load()
  enhancer.pickPreset('h3')
  enhancer.model = 'me/custom'
  enhancer.pickPreset('h3')
  expect(enhancer.model).toBe('me/custom')
})

it('reports a download that failed on the server', async () => {
  vi.useFakeTimers()
  api.listDownloads.mockResolvedValue({
    downloads: [{ repo_id: 'org/h3', status: 'failed', error: 'disk full' }],
  } as never)
  const enhancer = new Enhancer()
  enhancer.model = 'org/h3'
  const done = enhancer.downloadModel()
  expect(enhancer.downloading).toBe(true)
  await vi.advanceTimersByTimeAsync(2000)
  await done
  expect(enhancer.downloading).toBe(false)
  // our words lead; the server's reason is kept for Details
  expect(enhancer.error).toBe('The model download failed')
  expect(enhancer.errorDetail).toBe('disk full')
})

it('a refused enhancement leads with the described title, not the raw message', async () => {
  const raw = 'Traceback (most recent call last): KeyError: preset'
  api.enhance.mockRejectedValue(new ApiError(raw, 500))
  const enhancer = new Enhancer()
  enhancer.idea = 'a cat'
  await enhancer.generate()
  expect(enhancer.error).toBe(
    "Couldn't start the enhancement: The server hit an error",
  )
  expect(enhancer.error).not.toContain('Traceback')
  expect(enhancer.errorDetail).toBe(raw)
  expect(enhancer.busy).toBe(false)
})

it('hands the result over once, with what made it', () => {
  const enhancer = new Enhancer()
  enhancer.model = 'org/h3'
  enhancer.idea = 'a cat'
  enhancer.result = 'a tabby cat on a sill'
  expect(enhancer.takeResult()).toEqual({
    text: 'a tabby cat on a sill',
    enhanced: { model: 'org/h3', idea: 'a cat' },
  })
  expect(enhancer.result).toBe('')
})
