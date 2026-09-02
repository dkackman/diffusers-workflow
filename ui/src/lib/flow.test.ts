import { describe, expect, it } from 'vitest'
import { danglingReferenceDetails, dataFlowGraph, flowGraph } from './flow'

const step = (name: string, args: Record<string, unknown>) => ({
  name,
  task: { command: 'noop', arguments: args },
})

const pipelineStep = (name: string, args: Record<string, unknown>) => ({
  name,
  pipeline: {
    configuration: { component_type: 'ZImagePipeline' },
    arguments: args,
  },
})

describe('flowGraph', () => {
  it('builds fan-in: three generators feeding one combiner, not each other', () => {
    const wf = {
      steps: [
        step('gen1', {}),
        step('gen2', {}),
        step('gen3', {}),
        step('video', {
          a: 'previous_result:gen1',
          b: 'previous_result:gen2',
          c: 'previous_result:gen3',
        }),
      ],
    }
    const graph = flowGraph(wf)
    expect(graph[0]).toEqual({
      name: 'gen1',
      inputs: [],
      consumers: ['video'],
      resolvedRefs: 0,
    })
    expect(graph[1].consumers).toEqual(['video'])
    expect(graph[3].inputs).toEqual(['gen1', 'gen2', 'gen3'])
    expect(graph[3].resolvedRefs).toBe(3)
  })

  it('builds fan-out: one producer with two consumers', () => {
    const wf = {
      steps: [
        step('gen', {}),
        step('up', { image: 'previous_result:gen' }),
        step('caption', { image: 'previous_result:gen' }),
      ],
    }
    const graph = flowGraph(wf)
    expect(graph[0].consumers).toEqual(['up', 'caption'])
  })

  it('resolves media suffixes to the base step and finds refs in nested values', () => {
    const wf = {
      steps: [
        step('gen', {}),
        step('mux', {
          nested: { list: ['previous_result:gen.frames'] },
          audio: 'previous_result:gen.audio',
        }),
      ],
    }
    const graph = flowGraph(wf)
    expect(graph[1].inputs).toEqual(['gen'])
    expect(graph[1].resolvedRefs).toBe(2)
  })

  it('only earlier steps are producers - a later or missing name is no edge', () => {
    const wf = {
      steps: [step('a', { x: 'previous_result:b' }), step('b', {})],
    }
    const graph = flowGraph(wf)
    expect(graph[0].inputs).toEqual([])
    expect(graph[1].consumers).toEqual([])
  })

  it('dedupes repeated references to the same producer from different argument keys', () => {
    const wf = {
      steps: [
        step('gen', {}),
        step('x', {
          a: 'previous_result:gen',
          b: 'previous_result:gen',
        }),
      ],
    }
    const graph = flowGraph(wf)
    expect(graph[0].consumers).toEqual(['x'])
    expect(graph[1].resolvedRefs).toBe(2)
  })
})

describe('danglingReferenceDetails', () => {
  it('attributes each problem to its step index', () => {
    const wf = {
      variables: {},
      steps: [
        step('a', { x: 'variable:missing' }),
        step('b', { y: 'previous_result:nope' }),
      ],
    }
    const details = danglingReferenceDetails(wf)
    expect(details).toHaveLength(2)
    expect(details[0].stepIndex).toBe(0)
    expect(details[0].message).toContain('variable:missing')
    expect(details[1].stepIndex).toBe(1)
    expect(details[1].message).toContain('previous_result:nope')
  })
})

describe('dataFlowGraph', () => {
  it('identifies previous_result edges labeled with the attribute they feed', () => {
    const wf = {
      steps: [
        pipelineStep('gen', { prompt: 'variable:prompt' }),
        step('upscale', { image: 'previous_result:gen' }),
      ],
    }
    const graph = dataFlowGraph(wf)
    expect(graph.edges).toEqual([
      { from: 'gen', to: 'upscale', attribute: 'image' },
    ])
  })

  it('flags steps with no previous_result input as entry points, others not', () => {
    const wf = {
      steps: [
        pipelineStep('gen', { prompt: 'variable:prompt' }),
        step('upscale', { image: 'previous_result:gen' }),
      ],
    }
    const graph = dataFlowGraph(wf)
    expect(graph.nodes.find((n) => n.name === 'gen')?.isEntryPoint).toBe(true)
    expect(graph.nodes.find((n) => n.name === 'upscale')?.isEntryPoint).toBe(
      false,
    )
  })

  it('reports step kind and detail from pipeline/task/workflow shape', () => {
    const wf = {
      steps: [
        pipelineStep('gen', {}),
        step('caption', {}),
        { name: 'sub', workflow: { path: 'other.json', arguments: {} } },
      ],
    }
    const graph = dataFlowGraph(wf)
    expect(graph.nodes[0]).toMatchObject({
      kind: 'pipeline',
      detail: 'ZImagePipeline',
    })
    expect(graph.nodes[1]).toMatchObject({ kind: 'task', detail: 'noop' })
    expect(graph.nodes[2]).toMatchObject({
      kind: 'workflow',
      detail: 'other.json',
    })
  })

  it('flags fan-in when a step combines more than one upstream producer', () => {
    const wf = {
      steps: [
        pipelineStep('images', {}),
        pipelineStep('masks', {}),
        step('combine', {
          a: 'previous_result:images',
          b: 'previous_result:masks',
        }),
      ],
    }
    const graph = dataFlowGraph(wf)
    expect(graph.fanIn.has('combine')).toBe(true)
    expect(graph.fanIn.get('combine')?.producers.sort()).toEqual([
      'images',
      'masks',
    ])
    // Single-producer steps are not fan-in, even with multiple edges to it
    const single = dataFlowGraph({
      steps: [
        pipelineStep('gen', {}),
        step('mux', {
          frames: 'previous_result:gen.frames',
          audio: 'previous_result:gen.audio',
        }),
      ],
    })
    expect(single.fanIn.has('mux')).toBe(false)
  })

  it('derives a static multiplier from literal num_images_per_prompt on both producers', () => {
    const wf = {
      steps: [
        pipelineStep('images', { num_images_per_prompt: 4 }),
        pipelineStep('masks', { num_images_per_prompt: 3 }),
        step('combine', {
          a: 'previous_result:images',
          b: 'previous_result:masks',
        }),
      ],
    }
    const graph = dataFlowGraph(wf)
    expect(graph.fanIn.get('combine')?.label).toBe('4 × 3 = 12')
  })

  it('falls back to a structural fan-in label when counts are not statically known', () => {
    const wf = {
      steps: [
        pipelineStep('images', {}),
        pipelineStep('masks', {}),
        step('combine', {
          a: 'previous_result:images',
          b: 'previous_result:masks',
        }),
      ],
    }
    const graph = dataFlowGraph(wf)
    expect(graph.fanIn.get('combine')?.label).toBe('combines 2 upstream steps')
  })
})
