import { describe, expect, it } from 'vitest'
import { groupResultFiles } from './results'
import type { JobEvent } from './types'

const stepEnd = (step: string, files: string[]): JobEvent =>
  ({ seq: 0, event: 'step_end', step, files }) as unknown as JobEvent

describe('groupResultFiles', () => {
  it('groups streamed files by producing step, in completion order', () => {
    const groups = groupResultFiles(undefined, [
      stepEnd('generate', ['a.png', 'b.png']),
      stepEnd('upscale', ['a_big.png']),
    ])
    expect(groups).toEqual([
      { step: 'generate', files: ['a.png', 'b.png'] },
      { step: 'upscale', files: ['a_big.png'] },
    ])
  })

  it('merges the manifest without duplicating streamed files', () => {
    const groups = groupResultFiles(
      [{ step: 'generate', files: ['a.png', 'c.png'] }],
      [stepEnd('generate', ['a.png'])],
    )
    expect(groups).toEqual([{ step: 'generate', files: ['a.png', 'c.png'] }])
  })

  it('drops steps with no files and handles a historical job (manifest only)', () => {
    const groups = groupResultFiles(
      [
        { step: 'load', files: [] },
        { step: 'generate', files: ['a.png'] },
      ],
      [],
    )
    expect(groups).toEqual([{ step: 'generate', files: ['a.png'] }])
  })
})
