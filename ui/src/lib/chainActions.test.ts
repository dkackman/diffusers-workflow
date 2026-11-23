import { describe, expect, it } from 'vitest'
import {
  FIRST_SEGMENT,
  NO_REPAIR_TEMPLATE,
  NO_SEGMENTS,
  REPAIRED_TAKE,
  REPAIR_TEMPLATE,
  chainModel,
  continueAction,
  continueConfirmText,
  continueValidation,
  repairAction,
  repairModels,
} from './chainActions'
import { chainRows, type ChainTable } from './chainRows'
import type { ManifestEntry } from './types'

const H3 = 'MiniMaxAI/MiniMax-H3'
const LTX = 'Lightricks/LTX-2.5-Diffusers'

// The repair template as the catalog holds it: one pipeline step loading H3
const repairTemplate = {
  variables: { chain_take: '', segment: 2 },
  steps: [
    { name: 'split', task: { command: 'get_frame' } },
    {
      name: 'repair',
      pipeline: { from_pretrained_arguments: { model_name: H3 } },
    },
  ],
}

const shot = (i: number, extra: Record<string, unknown> = {}) => ({
  name: `segment ${i}`,
  start_frame: (i - 1) * 121,
  num_frames: 121,
  planned_segments: 4,
  ...extra,
})

const kept = (i: number) => shot(i, { segment_file: `take.segment-00${i}.mp4` })

const table = (shots: any[]): ChainTable =>
  chainRows({
    step: 'take',
    files: [
      'run/take.mp4',
      ...shots
        .filter((s) => s.segment_file)
        .map((s) => `run/${s.segment_file}`),
    ],
    shots,
  } as unknown as ManifestEntry) as ChainTable

const definition = (model: unknown, chain: Record<string, unknown> = {}) => ({
  variables: {
    model: H3,
    stop_after: 2,
    resume_from: null,
    keep_segments: true,
  },
  steps: [
    {
      name: 'take',
      pipeline: {
        from_pretrained_arguments: { model_name: model },
        chain,
      },
    },
  ],
})

describe('chainModel and repairModels', () => {
  it('reads the chain step model, following a variable', () => {
    expect(chainModel(definition(H3), 'take')).toBe(H3)
    expect(chainModel(definition('variable:model'), 'take')).toBe(H3)
    expect(
      chainModel(definition('variable:model'), 'take', { model: LTX }),
    ).toBe(LTX)
    expect(chainModel(definition(H3), 'other')).toBeNull()
    expect(chainModel(null, 'take')).toBeNull()
  })

  it('lists the models a pipeline step of the repair template loads', () => {
    expect(repairModels(repairTemplate)).toEqual([H3])
    expect(repairModels(null)).toEqual([])
  })
})

describe('repairAction', () => {
  const models = repairModels(repairTemplate)
  const t = table([kept(1), kept(2), kept(3)])

  it('opens repair-segment with the take and the segment number', () => {
    expect(repairAction(t, t.rows[2], H3, models)).toEqual({
      enabled: true,
      workflow: REPAIR_TEMPLATE,
      arguments: { chain_take: 'output:run/take.mp4', segment: '3' },
    })
  })

  it('is off for a family with no repair template', () => {
    expect(repairAction(t, t.rows[1], LTX, models)).toEqual({
      enabled: false,
      reason: NO_REPAIR_TEMPLATE,
    })
    expect(repairAction(t, t.rows[1], null, models)).toMatchObject({
      reason: NO_REPAIR_TEMPLATE,
    })
    // The template could not be read: nothing to repair with
    expect(repairAction(t, t.rows[1], H3, [])).toMatchObject({
      reason: NO_REPAIR_TEMPLATE,
    })
  })

  it('is off on segment 1', () => {
    expect(repairAction(t, t.rows[0], H3, models)).toEqual({
      enabled: false,
      reason: FIRST_SEGMENT,
    })
  })

  it('is off on every row of a repaired take', () => {
    const r = table([
      shot(1),
      shot(2, { name: 'segment 2 (repaired)' }),
      shot(3),
    ])
    expect(repairAction(r, r.rows[2], H3, models)).toEqual({
      enabled: false,
      reason: REPAIRED_TAKE,
    })
  })

  it('is off when the run did not keep its segments', () => {
    const r = table([shot(1), shot(2), shot(3)])
    expect(repairAction(r, r.rows[1], H3, models)).toEqual({
      enabled: false,
      reason: NO_SEGMENTS,
    })
  })

  it('gives exactly one reason, family first', () => {
    const r = table([shot(1), shot(2, { name: 'segment 2 (repaired)' })])
    expect(repairAction(r, r.rows[0], LTX, models)).toMatchObject({
      reason: NO_REPAIR_TEMPLATE,
    })
    expect(repairAction(r, r.rows[0], H3, models)).toMatchObject({
      reason: FIRST_SEGMENT,
    })
  })
})

describe('continueAction', () => {
  const both = {
    stop_after: 'variable:stop_after',
    resume_from: 'variable:resume_from',
  }

  it('resumes from the last kept segment and clears stop_after', () => {
    const t = table([kept(1), kept(2)])
    expect(continueAction(t, definition(H3, both))).toEqual({
      arguments: {
        resume_from: 'output:run/take.segment-002.mp4',
        stop_after: null,
      },
    })
  })

  it('needs both variables declared', () => {
    const t = table([kept(1), kept(2)])
    expect(
      continueAction(
        t,
        definition(H3, { resume_from: 'variable:resume_from' }),
      ),
    ).toBeNull()
    expect(
      continueAction(t, definition(H3, { stop_after: 'variable:stop_after' })),
    ).toBeNull()
    // A literal stop_after names no variable to clear
    expect(
      continueAction(
        t,
        definition(H3, { stop_after: 2, resume_from: 'variable:resume_from' }),
      ),
    ).toBeNull()
    // A variable the workflow does not declare
    expect(
      continueAction(
        t,
        definition(H3, { ...both, stop_after: 'variable:undeclared' }),
      ),
    ).toBeNull()
  })

  it('is not offered on a complete take', () => {
    const t = table([1, 2, 3, 4].map(kept))
    expect(continueAction(t, definition(H3, both))).toBeNull()
  })

  it('is not offered when a row has no segment file', () => {
    const t = table([kept(1), shot(2)])
    expect(continueAction(t, definition(H3, both))).toBeNull()
  })

  it('is offered on any family that declares both variables', () => {
    // Continue is not H3-only: an LTX chain with both variables offers it
    const t = table([kept(1), kept(2)])
    expect(continueAction(t, definition(LTX, both))).not.toBeNull()
  })
})

describe('continueValidation and continueConfirmText', () => {
  const action = {
    arguments: {
      resume_from: 'output:run/take.segment-002.mp4',
      stop_after: null,
    },
  }
  const ran = { workspace: 'ws1', arguments: { stop_after: 2, seed: 7 } }

  it('prices the rerun by the catalog name, with the merged arguments', () => {
    expect(
      continueValidation(
        { ...ran, workflow_name: 'chain-1451' },
        action,
        definition(LTX),
      ),
    ).toEqual({
      workflow_path: 'chain-1451',
      workspace: 'ws1',
      arguments: {
        seed: 7,
        stop_after: null,
        resume_from: 'output:run/take.segment-002.mp4',
      },
      sizes: false,
    })
  })

  it('falls back to the definition the job ran, inline', () => {
    const d = definition(LTX)
    expect(
      continueValidation({ ...ran, workflow_name: null }, action, d),
    ).toMatchObject({ workflow: d, workspace: 'ws1' })
  })

  it('has nothing to price with neither', () => {
    expect(
      continueValidation({ ...ran, workflow_name: null }, action, null),
    ).toBeNull()
  })

  it('states the rerun, then the estimate', () => {
    const text = continueConfirmText(
      table([kept(1), kept(2)]),
      'about 3.2 min on mps',
    )
    expect(text).toContain('(2 of 4)')
    expect(text).toContain('remaining 2 segments of take')
    expect(text).toMatch(/\n\nEstimate: about 3\.2 min on mps$/)
  })
})
