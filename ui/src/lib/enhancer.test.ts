import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { Enhancer } from './enhancer.svelte'
import { api } from './api'
import type { EnhancerPreset } from './types'

vi.mock('./api', () => ({
  api: {
    listEnhancers: vi.fn(),
    listModels: vi.fn().mockResolvedValue({ repos: [] }),
    startDownload: vi.fn().mockResolvedValue({}),
    listDownloads: vi.fn(),
  },
  fetchOutputText: vi.fn(),
  streamJobEvents: vi.fn(() => () => {}),
}))

const preset = (key: string, intended: string[] = []): EnhancerPreset => ({
  key,
  label: key,
  default_model: `org/${key}`,
  models: [`org/${key}`],
  intended_models: intended,
  placeholder: '',
})

beforeEach(() => {
  vi.mocked(api.listEnhancers).mockResolvedValue({
    presets: [preset('h3', ['minimax-h3']), preset('ltx', ['ltx-2'])],
  })
})
afterEach(() => vi.useRealTimers())

it('loads the presets, and says so when the server has none', async () => {
  const enhancer = new Enhancer()
  await enhancer.load()
  expect(enhancer.presets.map((p) => p.key)).toEqual(['h3', 'ltx'])
  expect(enhancer.down).toBe(false)

  vi.mocked(api.listEnhancers).mockResolvedValue({ presets: [] })
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
  vi.mocked(api.listDownloads).mockResolvedValue({
    downloads: [{ repo_id: 'org/h3', status: 'failed', error: 'disk full' }],
  } as never)
  const enhancer = new Enhancer()
  enhancer.model = 'org/h3'
  const done = enhancer.downloadModel()
  expect(enhancer.downloading).toBe(true)
  await vi.advanceTimersByTimeAsync(2000)
  await done
  expect(enhancer.downloading).toBe(false)
  expect(enhancer.error).toBe('disk full')
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
