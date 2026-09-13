import { describe, expect, it } from 'vitest'
import {
  groupResultFiles,
  sectionBySubfolder,
  unsavedReason,
  unsavedSteps,
} from './results'
import type { JobEvent } from './types'

const stepEnd = (
  step: string,
  files: string[],
  subfolder?: string,
  reused?: boolean,
): JobEvent =>
  ({
    seq: 0,
    event: 'step_end',
    step,
    files,
    subfolder,
    reused,
  }) as unknown as JobEvent

describe('groupResultFiles', () => {
  it('groups streamed files by producing step, in completion order', () => {
    const groups = groupResultFiles(undefined, [
      stepEnd('generate', ['a.png', 'b.png']),
      stepEnd('upscale', ['a_big.png']),
    ])
    expect(groups).toEqual([
      {
        step: 'generate',
        files: ['a.png', 'b.png'],
        reused: false,
        subfolder: '',
      },
      { step: 'upscale', files: ['a_big.png'], reused: false, subfolder: '' },
    ])
  })

  it('merges the manifest without duplicating streamed files', () => {
    const groups = groupResultFiles(
      [{ step: 'generate', files: ['a.png', 'c.png'] }],
      [stepEnd('generate', ['a.png'])],
    )
    expect(groups).toEqual([
      {
        step: 'generate',
        files: ['a.png', 'c.png'],
        reused: false,
        subfolder: '',
      },
    ])
  })

  it('drops steps with no files and handles a historical job (manifest only)', () => {
    const groups = groupResultFiles(
      [
        { step: 'load', files: [] },
        { step: 'generate', files: ['a.png'] },
      ],
      [],
    )
    expect(groups).toEqual([
      { step: 'generate', files: ['a.png'], reused: false, subfolder: '' },
    ])
  })

  it('marks a step the manifest says was served from the step cache', () => {
    const groups = groupResultFiles(
      [
        { step: 'generate', files: ['a.png'], reused: true },
        { step: 'upscale', files: ['a_big.png'] },
      ],
      [],
    )
    expect(groups.map((g) => g.reused)).toEqual([true, false])
  })

  it('reads reused live from step_end before any manifest arrives', () => {
    const groups = groupResultFiles(undefined, [
      stepEnd('generate', ['a.png'], undefined, true),
    ])
    expect(groups[0].reused).toBe(true)
  })
})

describe('groupResultFiles subfolder', () => {
  it('places a group from the live step_end and lets the manifest confirm it', () => {
    const groups = groupResultFiles(
      [{ step: 'episode', files: ['final/e.mp4'], subfolder: 'final' }],
      [stepEnd('episode', ['final/e.mp4'], 'final')],
    )
    expect(groups).toEqual([
      {
        step: 'episode',
        files: ['final/e.mp4'],
        reused: false,
        subfolder: 'final',
      },
    ])
  })

  it('reads the subfolder from the stream alone while the job runs', () => {
    const groups = groupResultFiles(undefined, [
      stepEnd('shot', ['intermediate/s.mp4'], 'intermediate'),
    ])
    expect(groups[0].subfolder).toBe('intermediate')
  })

  it('treats a manifest entry without the field as the run root', () => {
    const groups = groupResultFiles([{ step: 'old', files: ['a.png'] }], [])
    expect(groups[0].subfolder).toBe('')
  })

  it('lets the manifest win when it names a different subfolder', () => {
    const groups = groupResultFiles(
      [{ step: 's', files: ['final/x.png'], subfolder: 'final' }],
      [stepEnd('s', ['final/x.png'], '')],
    )
    expect(groups[0].subfolder).toBe('final')
  })
})

describe('sectionBySubfolder', () => {
  const group = (step: string, subfolder: string) => ({
    step,
    files: [`${subfolder ? subfolder + '/' : ''}${step}.png`],
    reused: false,
    subfolder,
  })

  it('returns one root section when no group has a subfolder', () => {
    const sections = sectionBySubfolder([group('a', ''), group('b', '')])
    expect(sections).toEqual([
      { subfolder: '', groups: [group('a', ''), group('b', '')] },
    ])
  })

  it('returns one root section for no groups at all', () => {
    expect(sectionBySubfolder([])).toEqual([{ subfolder: '', groups: [] }])
  })

  it('sections by subfolder in first-appearance order, final first', () => {
    const sections = sectionBySubfolder([
      group('draw', 'intermediate'),
      group('shot', 'intermediate'),
      group('episode', 'final'),
      group('notes', ''),
    ])
    expect(sections.map((s) => s.subfolder)).toEqual([
      'final',
      'intermediate',
      '',
    ])
    expect(sections[1].groups.map((g) => g.step)).toEqual(['draw', 'shot'])
  })

  it('keeps any name the engine accepted, not just the convention', () => {
    const sections = sectionBySubfolder([
      group('a', 'shots/act-1'),
      group('b', 'final'),
    ])
    expect(sections.map((s) => s.subfolder)).toEqual(['final', 'shots/act-1'])
  })
})

describe('unsavedReason', () => {
  it('names result.save when the workflow asked for no file', () => {
    expect(
      unsavedReason({ result: { save: false, content_type: 'image/png' } }),
    ).toEqual({
      key: 'result.save',
      detail: 'is false, so the step is kept in memory and never written',
    })
  })

  it('names result.content_type when there is no type to write', () => {
    expect(unsavedReason({ task: { command: 'mux' } })).toEqual({
      key: 'result.content_type',
      detail: 'is not declared, so there is no file type to write',
    })
  })

  it('has no reason for a step that declares a file', () => {
    expect(
      unsavedReason({ result: { content_type: 'video/mp4', save: true } }),
    ).toBeNull()
  })
})

describe('unsavedSteps', () => {
  const definition = (steps: Array<Record<string, unknown>>) => ({ steps })
  const steps = definition([
    { name: 'base', result: { save: false } },
    { name: 'edit', task: { command: 'mux' } },
    { name: 'film', result: { content_type: 'video/mp4' } },
  ])

  it('names every step that ran and wrote nothing, with the reason', () => {
    const unsaved = unsavedSteps(
      [
        { step: 'base', files: [] },
        { step: 'edit', files: [] },
        { step: 'film', files: ['final/film.mp4'], subfolder: 'final' },
      ],
      [],
      steps,
    )
    expect(unsaved).toEqual([
      {
        node: 'base',
        members: [],
        reason: {
          key: 'result.save',
          detail: 'is false, so the step is kept in memory and never written',
        },
      },
      {
        node: 'edit',
        members: [],
        reason: {
          key: 'result.content_type',
          detail: 'is not declared, so there is no file type to write',
        },
      },
    ])
  })

  it('collapses a for_each group into the one step the workflow declares', () => {
    const unsaved = unsavedSteps(
      [
        { step: 'base@open', files: [] },
        { step: 'base@reveal', files: [] },
      ],
      [],
      steps,
    )
    expect(unsaved).toEqual([
      {
        node: 'base',
        members: ['base@open', 'base@reveal'],
        reason: {
          key: 'result.save',
          detail: 'is false, so the step is kept in memory and never written',
        },
      },
    ])
  })

  it('lists only steps that have finished, so a running job is not accused', () => {
    const events = [
      { seq: 1, event: 'workflow_start', steps: ['base', 'edit', 'film'] },
      { seq: 2, event: 'step_end', step: 'base', files: [] },
    ] as unknown as JobEvent[]
    expect(unsavedSteps(undefined, events, steps).map((s) => s.node)).toEqual([
      'base',
    ])
  })

  it("ignores a sub-workflow's inner steps, which are no node of this graph", () => {
    const composed = definition([
      { name: 'shot1_amnesty', workflow: { path: 'child.json' } },
    ])
    const unsaved = unsavedSteps(
      [
        { step: 'reference_to_video_audio', files: [] },
        { step: 'shot1_amnesty', files: ['intermediate/a.mp4'] },
      ],
      [],
      composed,
    )
    expect(unsaved).toEqual([])
  })

  it('has nothing to say without a definition to read the reasons off', () => {
    expect(unsavedSteps([{ step: 'base', files: [] }], [], null)).toEqual([])
  })
})
