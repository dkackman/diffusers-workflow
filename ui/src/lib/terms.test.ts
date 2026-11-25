import { describe, expect, it } from 'vitest'
import { ordinal, versionTitle, workflowName } from './terms'

describe('workflowName', () => {
  it('workflowName prefers the file name', () => {
    expect(
      workflowName({ workflow: 'E2ECard', workflow_name: 'e2e-card' }),
    ).toBe('e2e-card')
    expect(
      workflowName({
        workflow: 'RefToVideo',
        workflow_name: 'templates/minimax/ref-to-video',
      }),
    ).toBe('ref-to-video')
  })
  it('falls back to the JSON id, then to "job"', () => {
    expect(workflowName({ workflow: 'minimax/wf', workflow_name: null })).toBe(
      'wf',
    )
    expect(workflowName({ workflow: null, workflow_name: null })).toBe('job')
    expect(workflowName({ workflow_name: 'dir/' })).toBe('job')
  })
})

describe('ordinal', () => {
  it.each([
    [1, '1st'],
    [2, '2nd'],
    [3, '3rd'],
    [4, '4th'],
    [11, '11th'],
    [12, '12th'],
    [13, '13th'],
    [21, '21st'],
    [22, '22nd'],
    [111, '111th'],
  ])('%i is %s', (n, want) => {
    expect(ordinal(n)).toBe(want)
  })
  it('titles the version badge', () => {
    expect(versionTitle(2)).toBe('2nd job of this workflow in this folder')
  })
})
