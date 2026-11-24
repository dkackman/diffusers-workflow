import { expect, it } from 'vitest'
import { recipeRows, savedFrom } from './recipe'
import type { GalleryMetadata, JobDetail } from './types'

const meta = (fields: Partial<GalleryMetadata>): GalleryMetadata => ({
  name: 'portraits/r1/a.png',
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

const job = (fields: Partial<JobDetail>): JobDetail =>
  ({
    id: 'j9',
    status: 'succeeded',
    arguments: {},
    workflow: 'clip-id',
    workflow_name: 'video/clip',
    ...fields,
  }) as JobDetail

it("an image's rows come from its embedded recipe, in order", () => {
  const rows = recipeRows(
    meta({
      metadata: {
        workflow: { id: 'portrait' },
        step_name: 'draw',
        model_name: 'org/model',
        seed: 42,
        arguments: {
          prompt: 'a cat',
          negative_prompt: 'blur',
          seed: 42,
          steps: 20,
        },
      },
      job: { id: 'j1', status: 'succeeded' },
      media: { kind: 'image', width: 1024, height: 768 },
    }),
    null,
    'studio',
  )
  expect(rows.map((r) => [r.label, r.value])).toEqual([
    ['Workflow', 'portrait'],
    ['Job', 'j1'],
    ['Step', 'draw'],
    ['Model', 'org/model'],
    ['Seed', '42'],
    ['Prompt', 'a cat'],
    ['Negative prompt', 'blur'],
    ['Size', '1024 × 768'],
    ['steps', '20'],
  ])
  expect(rows.filter((r) => r.copy).map((r) => r.key)).toEqual([
    'seed',
    'prompt',
    'negative',
  ])
})

it("a clip's rows come from its job's arguments", () => {
  const rows = recipeRows(
    meta({
      job: { id: 'j9', status: 'succeeded' },
      media: { kind: 'video', duration_seconds: 75 },
    }),
    job({ arguments: { prompt: 'waves', seed: 7, frames: 121 } }),
    'studio',
  )
  expect(rows.map((r) => [r.key, r.label, r.value])).toEqual([
    ['workflow', 'Workflow', 'clip'],
    ['job', 'Job', 'j9'],
    ['seed', 'Seed', '7'],
    ['prompt', 'Prompt', 'waves'],
    ['length', 'Length', '1m 15s'],
    ['argument', 'frames', '121'],
  ])
})

it('the Workflow row is the job file name, not the embedded id', () => {
  const rows = recipeRows(
    meta({
      metadata: { workflow: { id: 'portrait-json-id' } },
      job: { id: 'j9', status: 'succeeded' },
    }),
    job({ workflow_name: 'people/portrait' }),
    'studio',
  )
  expect(rows.find((r) => r.key === 'workflow')?.value).toBe('portrait')
  const forgotten = recipeRows(
    meta({
      metadata: { workflow: { id: 'portrait-json-id' } },
      job: { id: 'j9', status: 'forgotten' },
    }),
    null,
    'studio',
  )
  expect(forgotten.find((r) => r.key === 'workflow')?.value).toBe(
    'portrait-json-id',
  )
})

it('every argument is listed once', () => {
  const rows = recipeRows(
    meta({
      metadata: {
        arguments: {
          prompt: 'a cat',
          seed: 3,
          guidance: { scale: 4 },
          width: 512,
        },
      },
    }),
    null,
    'studio',
  )
  const args = rows.filter((r) => r.key === 'argument')
  expect(args.map((r) => [r.label, r.value])).toEqual([
    ['guidance', '{"scale":4}'],
    ['width', '512'],
  ])
  expect(rows.filter((r) => r.value === 'a cat')).toHaveLength(1)
  expect(rows.filter((r) => r.value === '3')).toHaveLength(1)
})

it('the job links its own workspace', () => {
  const rows = recipeRows(
    meta({ job: { id: 'j1', status: 'succeeded' } }),
    null,
    'elsewhere',
  )
  expect(rows.find((r) => r.key === 'job')?.href).toBe('#/ws/elsewhere/jobs/j1')
})

it('an asset saved from a job says so', () => {
  expect(
    savedFrom(
      meta({
        source: 'asset',
        job: {
          id: 'j1',
          status: 'succeeded',
          workflow_name: 'people/portrait',
        },
        version: 3,
      }),
    ),
  ).toBe('Saved from portrait v3')
  expect(
    savedFrom(
      meta({ source: 'asset', job: { id: 'j2', status: 'forgotten' } }),
    ),
  ).toBe('Saved from j2')
  expect(savedFrom(meta({ source: 'asset' }))).toBe('')
})
