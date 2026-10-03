import { describe, expect, it } from 'vitest'
import { danglingReferenceDetails, dataFlowGraph, flowGraph } from './flow'
// Read by tests/test_ui_twins.py too, where the engine decides each case
import referenceCases from '../../../tests/fixtures/reference_cases.json'

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
  const forEachStep = (name: string) => ({
    ...step(name, { x: 'item:prompt' }),
    for_each: [{ name: 'a', prompt: 'p' }],
  })

  it('accepts a member reference to a for_each step, with or without a property', () => {
    const wf = {
      variables: {},
      steps: [
        forEachStep('g'),
        step('b', {
          one: 'previous_result:g@a',
          two: 'previous_result:g@a.mask',
        }),
      ],
    }
    expect(danglingReferenceDetails(wf)).toEqual([])
  })

  it('flags a member reference to a step that is not for_each', () => {
    const wf = {
      variables: {},
      steps: [step('g', {}), step('b', { y: 'previous_result:g@a' })],
    }
    expect(danglingReferenceDetails(wf)).toHaveLength(1)
  })

  it('matches a dotted step name whole, and a property of a plain one', () => {
    const wf = {
      variables: {},
      steps: [
        step('x.y', {}),
        step('seg', {}),
        step('b', { v: 'previous_result:x.y', m: 'previous_result:seg.mask' }),
      ],
    }
    expect(danglingReferenceDetails(wf)).toEqual([])
  })

  it('checks a from_previous_result spelled as a reference as that reference', () => {
    // The engine leaves a variable: under this key to variable resolution
    const wf = {
      variables: { source: 'a' },
      steps: [
        step('a', {}),
        step('b', { image: { from_previous_result: 'variable:source' } }),
        step('c', { image: { from_previous_result: 'variable:missing' } }),
      ],
    }
    const details = danglingReferenceDetails(wf)
    expect(details).toHaveLength(1)
    expect(details[0].message).toContain('variable:missing')
  })

  it('flags a from_previous_result that names no earlier step', () => {
    const wf = {
      variables: {},
      steps: [step('b', { image: { from_previous_result: 'nope' } })],
    }
    const details = danglingReferenceDetails(wf)
    expect(details).toHaveLength(1)
    expect(details[0].message).toContain('nope')
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

describe('for_each references', () => {
  const cut = {
    steps: [
      step('draw', {}),
      step('slice', { audio: 'previous_result:song' }),
      {
        name: 'shot',
        for_each: 'variable:shots',
        pipeline: {
          configuration: { component_type: 'ModularPipeline' },
          arguments: {
            prompt: 'item:prompt',
            references: [
              { reference_type: 'T', from_previous_result: 'draw' },
              { reference_type: 'T', from_previous_result: 'slice' },
            ],
          },
        },
      },
      step('edit', { videos: 'gather:shot' }),
    ],
  }

  it('from_previous_result inside a reference list is an edge labeled by the list', () => {
    const graph = dataFlowGraph(cut)
    expect(graph.edges).toContainEqual({
      from: 'draw',
      to: 'shot',
      attribute: 'references',
    })
    expect(graph.edges).toContainEqual({
      from: 'slice',
      to: 'shot',
      attribute: 'references',
    })
    expect(graph.nodes.find((n) => n.name === 'shot')?.isEntryPoint).toBe(false)
  })

  it('gather: is an edge from the for_each step to the step that gathers it', () => {
    const graph = dataFlowGraph(cut)
    expect(graph.edges).toContainEqual({
      from: 'shot',
      to: 'edit',
      attribute: 'videos',
    })
    expect(graph.nodes.find((n) => n.name === 'edit')?.isEntryPoint).toBe(false)
    expect(flowGraph(cut)[3].inputs).toEqual(['shot'])
    expect(flowGraph(cut)[2].consumers).toEqual(['edit'])
  })

  it('a gather of a step that does not exist is dangling', () => {
    const wf = { steps: [step('edit', { videos: 'gather:nope' })] }
    expect(danglingReferenceDetails(wf)).toEqual([
      {
        stepIndex: 0,
        message: "Step 'edit': gather:nope - names no earlier for_each step",
      },
    ])
  })
})

describe('for_each members', () => {
  it("lists a declared variable list's entry names as the step's members", () => {
    // What a realized workflow carries: for_each still names the variable,
    // and the run's actual list sits in variables (dw/realize.py)
    const wf = {
      variables: { shots: [{ name: 'open' }, { name: 'reveal' }] },
      steps: [
        {
          name: 'shot',
          for_each: 'variable:shots',
          task: { command: 'render' },
        },
      ],
    }
    const node = dataFlowGraph(wf).nodes[0]
    expect(node.forEach).toBe(true)
    expect(node.members).toEqual(['open', 'reveal'])
  })

  it('keys unnamed entries by index, the way the engine names members', () => {
    const wf = {
      variables: { items: ['a', 'b'] },
      steps: [
        { name: 'run', for_each: 'variable:items', task: { command: 'x' } },
      ],
    }
    expect(dataFlowGraph(wf).nodes[0].members).toEqual(['0', '1'])
  })

  it('reads a literal for_each list written on the step itself', () => {
    const wf = {
      steps: [
        {
          name: 'run',
          for_each: [{ name: 'a' }, { name: 'b' }],
          task: { command: 'x' },
        },
      ],
    }
    expect(dataFlowGraph(wf).nodes[0].members).toEqual(['a', 'b'])
  })

  it('marks a list-driven step whose list cannot be read, and leaves plain steps alone', () => {
    const wf = {
      variables: { other: 3 },
      steps: [
        {
          name: 'run',
          for_each: 'variable:missing',
          task: { command: 'x' },
        },
        step('plain', {}),
      ],
    }
    const graph = dataFlowGraph(wf)
    expect(graph.nodes[0].forEach).toBe(true)
    expect(graph.nodes[0].members).toBeNull()
    expect(graph.nodes[1].forEach).toBe(false)
    expect(graph.nodes[1].members).toBeNull()
  })
})

describe('the graph resolves references as the dangling check does', () => {
  const wf = {
    variables: {},
    steps: [
      step('x.y', {}),
      { ...step('g', { p: 'item:prompt' }), for_each: [{ name: 'a' }] },
      step('b', {
        whole: 'previous_result:x.y',
        member: 'previous_result:g@a.mask',
      }),
    ],
  }

  it('gives a dotted name and a member reference their producer chips', () => {
    expect(flowGraph(wf)[2].inputs.sort()).toEqual(['g', 'x.y'])
  })

  it('draws their data-flow edges', () => {
    const graph = dataFlowGraph(wf)
    expect(graph.edges.map((e) => e.from).sort()).toEqual(['g', 'x.y'])
    expect(graph.nodes[2].isEntryPoint).toBe(false)
  })
})

describe('the shared reference cases', () => {
  it.each(referenceCases)('$name', ({ workflow, flagged }) => {
    const messages = danglingReferenceDetails(workflow).map((d) => d.message)
    for (const fragment of flagged)
      expect(messages.some((m) => m.includes(fragment))).toBe(true)
    if (flagged.length === 0) expect(messages).toEqual([])
  })
})
