import { describe, expect, it } from 'vitest'
import { finishedNodes, flowNodeName } from './runstate'
import type { JobEvent } from './types'

const ended = (step: string, parent_step?: string): JobEvent =>
  ({ seq: 0, event: 'step_end', step, parent_step }) as JobEvent

describe('flowNodeName', () => {
  it('reduces a for_each member to the group the definition declares', () => {
    expect(flowNodeName('base@open')).toBe('base')
  })

  it("reduces a sub-workflow's inner step to the composed step that queued it", () => {
    expect(flowNodeName('reference_to_video_audio', 'shot1_amnesty')).toBe(
      'shot1_amnesty',
    )
  })

  it('takes the parent even when the child step is a for_each group itself', () => {
    expect(flowNodeName('base@open', 'shot@reveal')).toBe('shot')
  })

  it('leaves an ordinary step alone, and has nothing for a missing name', () => {
    expect(flowNodeName('film')).toBe('film')
    expect(flowNodeName(undefined)).toBeUndefined()
  })
})

describe('finishedNodes', () => {
  const members = ['base@open', 'base@reveal', 'film']

  it('finishes a for_each group only once every member has ended', () => {
    expect(finishedNodes([ended('base@open')], members)).toEqual([])
    expect(
      finishedNodes([ended('base@open'), ended('base@reveal')], members),
    ).toEqual(['base'])
  })

  it('finishes a plain step on its own end', () => {
    expect(finishedNodes([ended('film')], members)).toEqual(['film'])
  })

  it("does not finish a composed step for one of its child's inner steps", () => {
    const names = ['shot1_amnesty', 'episode']
    const events = [ended('inner_a', 'shot1_amnesty')]
    expect(finishedNodes(events, names)).toEqual([])
    events.push(ended('shot1_amnesty'))
    expect(finishedNodes(events, names)).toEqual(['shot1_amnesty'])
  })

  it('has nothing finished before the run has reported anything', () => {
    expect(finishedNodes([], members)).toEqual([])
  })
})
