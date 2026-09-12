import { describe, expect, it } from 'vitest'
import { groupResultFiles, sectionBySubfolder } from './results'
import type { JobEvent } from './types'

const stepEnd = (step: string, files: string[], subfolder?: string): JobEvent =>
  ({ seq: 0, event: 'step_end', step, files, subfolder }) as unknown as JobEvent

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
