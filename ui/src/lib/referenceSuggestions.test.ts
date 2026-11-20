import { describe, expect, it } from 'vitest'
import {
  danglingReferences,
  referenceSuggestions,
} from './referenceSuggestions'

describe('referenceSuggestions', () => {
  const workflow = {
    variables: { prompt: 'x', steps: 9 },
    steps: [
      { name: 'gen', result: { content_type: 'video/mp4' } },
      { name: 'refine' },
      { name: 'last' },
    ],
  }

  it('offers variables plus only earlier steps', () => {
    const forLast = referenceSuggestions(workflow, 2)
    expect(forLast).toContain('variable:prompt')
    expect(forLast).toContain('previous_result:gen')
    expect(forLast).toContain('previous_result:refine')
    expect(forLast).not.toContain('previous_result:last')

    const forFirst = referenceSuggestions(workflow, 0)
    expect(forFirst.filter((s) => s.startsWith('previous_result'))).toEqual([])
  })

  it('adds media suffixes for video-producing steps', () => {
    const forSecond = referenceSuggestions(workflow, 1)
    expect(forSecond).toContain('previous_result:gen.frames')
    expect(forSecond).toContain('previous_result:gen.audio')
    expect(forSecond).not.toContain('previous_result:refine.frames')
  })

  it('offers every stored prompt when the library is supplied', () => {
    const suggestions = referenceSuggestions(workflow, 0, [
      'scenic',
      'minimax/fox',
    ])
    expect(suggestions).toContain('prompt:scenic')
    expect(suggestions).toContain('prompt:minimax/fox')
    expect(referenceSuggestions(workflow, 0)).not.toContain('prompt:scenic')
  })
})

describe('referenceSuggestions for for_each steps', () => {
  const shots = [
    { name: 'open', prompt: 'a', references: [] },
    { name: 'close', prompt: 'b', seconds: 4 },
  ]
  const workflow = {
    variables: { shots },
    steps: [
      { name: 'shot', for_each: 'variable:shots', pipeline: {} },
      { name: 'grade', for_each: 'variable:shots', task: {} },
      { name: 'join', task: {} },
    ],
  }

  it('offers item: and every field an entry carries inside a for_each', () => {
    const inShot = referenceSuggestions(workflow, 0)
    expect(inShot).toEqual(
      expect.arrayContaining([
        'item:',
        'item:name',
        'item:prompt',
        'item:references',
        'item:seconds',
      ]),
    )
    expect(referenceSuggestions(workflow, 2)).not.toContain('item:')
  })

  it('offers gather: and members of an earlier for_each step', () => {
    const inJoin = referenceSuggestions(workflow, 2)
    expect(inJoin).toContain('gather:shot')
    expect(inJoin).toContain('previous_result:shot@open')
    expect(inJoin).toContain('previous_result:shot@close')
  })

  it('offers a plain previous_result: to it only from a for_each over the same list', () => {
    expect(referenceSuggestions(workflow, 1)).toContain('previous_result:shot')
    expect(referenceSuggestions(workflow, 2)).not.toContain(
      'previous_result:shot',
    )
  })

  it('suggests nothing the dangling check would flag', () => {
    for (const index of [1, 2]) {
      const offered = referenceSuggestions(workflow, index).filter(
        (s) => s.startsWith('previous_result:') || s.startsWith('gather:'),
      )
      const probe = structuredClone(workflow) as Record<string, any>
      probe.steps[index].task = { arguments: { value: offered } }
      expect(danglingReferences(probe)).toEqual([])
    }
  })
})

describe('danglingReferences', () => {
  it('flags undeclared variables and unknown or later steps', () => {
    const workflow = {
      variables: { prompt: 'x' },
      steps: [
        {
          name: 'gen',
          pipeline: {
            arguments: {
              prompt: 'variable:prompt',
              image: 'previous_result:later',
            },
          },
        },
        {
          name: 'later',
          pipeline: { arguments: { size: 'variable:missing' } },
        },
      ],
    }
    const problems = danglingReferences(workflow)
    expect(problems).toHaveLength(2)
    expect(problems[0]).toContain('previous_result:later')
    expect(problems[1]).toContain('variable:missing')
  })

  it('accepts valid references, property paths and nested values', () => {
    const workflow = {
      variables: { prompt: 'x' },
      steps: [
        { name: 'gen', result: { content_type: 'video/mp4' } },
        {
          name: 'use',
          task: {
            arguments: {
              frames: 'previous_result:gen.frames',
              nested: [{ deep: 'variable:prompt' }],
            },
          },
        },
      ],
    }
    expect(danglingReferences(workflow)).toEqual([])
  })

  it('checks prompt references only against a supplied library', () => {
    const workflow = {
      steps: [
        { name: 'gen', pipeline: { arguments: { prompt: 'prompt:scenic' } } },
      ],
    }
    // no listing - the server resolves these at run time, nothing to flag
    expect(danglingReferences(workflow)).toEqual([])
    expect(danglingReferences(workflow, ['scenic'])).toEqual([])
    const problems = danglingReferences(workflow, ['other'])
    expect(problems).toHaveLength(1)
    expect(problems[0]).toContain('prompt:scenic')
  })
})
