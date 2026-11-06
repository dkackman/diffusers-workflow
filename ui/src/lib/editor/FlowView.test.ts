import { render } from '@testing-library/svelte'
import { describe, expect, it } from 'vitest'
import FlowView from './FlowView.svelte'

const workflow = {
  steps: [
    { name: 'gen', pipeline: { configuration: { component_type: 'Fake' } } },
    {
      name: 'upscale',
      pipeline: { arguments: { image: 'previous_result:gen' } },
    },
  ],
}

function nodeFor(container: HTMLElement, name: string) {
  return [...container.querySelectorAll('g.node')].find((node) =>
    node.getAttribute('aria-label')?.startsWith(`step ${name},`),
  )!
}

it('marks nodes as controls only when a select handler is given', () => {
  const plain = render(FlowView, { workflow }).container
  expect(nodeFor(plain, 'gen').hasAttribute('role')).toBe(false)
  expect(nodeFor(plain, 'gen').hasAttribute('tabindex')).toBe(false)

  const clickable = render(FlowView, { workflow, onselect: () => {} }).container
  expect(nodeFor(clickable, 'gen').getAttribute('role')).toBe('button')
})

it('highlights the running step and the ones already finished', () => {
  const { container } = render(FlowView, {
    workflow,
    activeStep: 'upscale',
    doneSteps: ['gen'],
  })
  const gen = nodeFor(container, 'gen')
  const upscale = nodeFor(container, 'upscale')
  expect(gen.classList.contains('done')).toBe(true)
  expect(gen.classList.contains('active')).toBe(false)
  expect(upscale.classList.contains('active')).toBe(true)
  // the state is announced, not just drawn
  expect(gen.getAttribute('aria-label')).toContain('done')
  expect(upscale.getAttribute('aria-label')).toContain('active')
})

it('leaves every node unstyled when no run state is supplied', () => {
  const { container } = render(FlowView, { workflow })
  for (const node of container.querySelectorAll('g.node')) {
    expect(node.classList.contains('done')).toBe(false)
    expect(node.classList.contains('active')).toBe(false)
  }
})

it('keeps a long detail inside its box, showing the tail and the whole on hover', () => {
  const composed = {
    steps: [
      {
        name: 'shot1_inventory',
        workflow: { path: 'templates/minimax/composable-reference-shot.json' },
      },
      // a second step, so the graph draws boxes rather than one line
      { name: 'other', task: { command: 'save' } },
    ],
  }
  const { container } = render(FlowView, { workflow: composed })
  const node = nodeFor(container, 'shot1_inventory')
  const detail = node.querySelector('.stepdetail')!
  // A path is told apart by its end, so that is the part kept
  expect(detail.textContent).toMatch(/^…/)
  expect(detail.textContent).toMatch(/reference-shot\.json$/)
  expect(detail.textContent!.length).toBeLessThan(
    'templates/minimax/composable-reference-shot.json'.length,
  )
  expect(node.querySelector('title')?.textContent).toBe(
    'templates/minimax/composable-reference-shot.json',
  )
  // The box clips whatever a wider glyph set still pushes past its edge
  expect(node.getAttribute('clip-path')).toBe('url(#flow-nodebox)')
})

it('shortens a long step name from the end, leaving room for the entry tag', () => {
  const long = {
    steps: [
      {
        name: 'reference_to_video_audio_with_lipsync',
        pipeline: { configuration: { component_type: 'Fake' } },
      },
      { name: 'other', task: { command: 'save' } },
    ],
  }
  const { container } = render(FlowView, { workflow: long })
  const node = nodeFor(container, 'reference_to_video_audio_with_lipsync')
  const name = node.querySelector('.stepname')!
  expect(name.textContent).toMatch(/…$/)
  expect(name.textContent!.length).toBeLessThan(
    'reference_to_video_audio_with_lipsync'.length,
  )
  // A name that fits gets no tooltip at all
  expect(
    nodeFor(render(FlowView, { workflow }).container, 'gen').querySelector(
      'title',
    ),
  ).toBeNull()
  expect(node.querySelector('title')?.textContent).toBe(
    'reference_to_video_audio_with_lipsync',
  )
})

describe('for_each members', () => {
  const listWorkflow = {
    variables: { shots: [{ name: 'open' }, { name: 'reveal' }] },
    steps: [
      {
        name: 'shot',
        for_each: 'variable:shots',
        pipeline: { configuration: { component_type: 'Fake' } },
      },
      { name: 'episode', task: { command: 'mux' } },
    ],
  }

  function memberFor(container: HTMLElement, full: string) {
    return [...container.querySelectorAll('g.member')].find((m) =>
      m.getAttribute('aria-label')?.startsWith(`member ${full},`),
    )!
  }

  it("renders a list step's entries as inset chips in run order, and plain steps without any", () => {
    const { container } = render(FlowView, { workflow: listWorkflow })
    const shot = nodeFor(container, 'shot')
    expect(shot.getAttribute('aria-label')).toContain('for_each with 2 entries')
    const chips = [...shot.querySelectorAll('g.member')]
    expect(chips.map((c) => c.querySelector('text')?.textContent)).toEqual([
      'open',
      'reveal',
    ])
    // The whole engine name is on the chip, as hover text
    expect(chips[0].querySelector('title')?.textContent).toBe('shot@open')
    expect(
      nodeFor(container, 'episode').querySelectorAll('g.member'),
    ).toHaveLength(0)
    expect(
      nodeFor(container, 'episode').getAttribute('aria-label'),
    ).not.toContain('for_each')
  })

  it('colors the chips a run reports done and running', () => {
    const { container } = render(FlowView, {
      workflow: listWorkflow,
      doneMembers: ['shot@open'],
      activeMember: 'shot@reveal',
    })
    expect(memberFor(container, 'shot@open').classList.contains('done')).toBe(
      true,
    )
    expect(
      memberFor(container, 'shot@reveal').classList.contains('active'),
    ).toBe(true)
    // and announces the state, as the group node does
    expect(memberFor(container, 'shot@reveal').getAttribute('aria-label')).toBe(
      'member shot@reveal, active',
    )
  })

  it('leaves chips unstyled when no run state is supplied', () => {
    const { container } = render(FlowView, { workflow: listWorkflow })
    for (const chip of container.querySelectorAll('g.member')) {
      expect(chip.classList.contains('done')).toBe(false)
      expect(chip.classList.contains('active')).toBe(false)
    }
  })

  it('grows the box to hold its chips, and keeps a plain box for the rest', () => {
    const { container } = render(FlowView, { workflow: listWorkflow })
    const height = (name: string) =>
      Number(
        nodeFor(container, name)
          .querySelector('rect.box')!
          .getAttribute('height'),
      )
    expect(height('shot')).toBeGreaterThan(height('episode'))
  })

  it('says for_each rather than nothing when the list cannot be read', () => {
    const unknown = {
      steps: [
        {
          name: 'shot',
          for_each: 'variable:missing',
          task: { command: 'render' },
        },
      ],
    }
    const { container } = render(FlowView, { workflow: unknown })
    const shot = nodeFor(container, 'shot')
    expect(shot.getAttribute('aria-label')).toContain('for_each')
    expect(shot.querySelectorAll('g.member')).toHaveLength(0)
    expect(shot.querySelector('.membersunknown')?.textContent).toBe('for_each')
  })
})

describe('edges', () => {
  // Job 16c15bbff9d9's shape: base feeds up, refine and mux, so two edges
  // skip a column, and base -> mux carries two arguments
  const skipping = {
    steps: [
      { name: 'base', pipeline: {} },
      {
        name: 'up',
        task: { arguments: { latents: 'previous_result:base.latents' } },
      },
      {
        name: 'refine',
        pipeline: {
          arguments: {
            latents: 'previous_result:up',
            hold_audio: 'previous_result:base.audio',
          },
        },
      },
      {
        name: 'mux',
        task: {
          arguments: {
            video: 'previous_result:refine.videos',
            audio: 'previous_result:base.audio',
            sample_rate: 'previous_result:base.sampling_rate',
          },
        },
      },
    ],
  }

  it('draws one line per pair of steps, labeled with every argument it feeds', () => {
    const { container } = render(FlowView, { workflow: skipping })
    expect(container.querySelectorAll('path.edge')).toHaveLength(5)
    const labels = [...container.querySelectorAll('text.edgelabel')].map(
      (t) => t.textContent,
    )
    expect(labels).toContain('audio, sample_rate')
    expect(labels).not.toContain('audio')
  })

  it('gives every label its own place, lane labels above the boxes', () => {
    const { container } = render(FlowView, { workflow: skipping })
    const labels = [...container.querySelectorAll('text.edgelabel')]
    const places = labels.map(
      (t) => `${t.getAttribute('x')},${t.getAttribute('y')}`,
    )
    expect(new Set(places).size).toBe(places.length)

    const nodeTop = Number(
      /translate\([^,]+, ([\d.]+)\)/.exec(
        nodeFor(container, 'up').getAttribute('transform')!,
      )![1],
    )
    const laneLabels = labels.filter((t) =>
      ['hold_audio', 'audio, sample_rate'].includes(t.textContent!),
    )
    expect(laneLabels).toHaveLength(2)
    for (const t of laneLabels)
      expect(Number(t.getAttribute('y'))).toBeLessThan(nodeTop)
  })

  it('keeps the whole argument list on hover when the label is cut', () => {
    const long = {
      steps: [
        { name: 'a', pipeline: {} },
        {
          name: 'b',
          pipeline: {
            arguments: {
              first_very_long_argument: 'previous_result:a',
              second_very_long_argument: 'previous_result:a',
            },
          },
        },
      ],
    }
    const { container } = render(FlowView, { workflow: long })
    const label = container.querySelector('text.edgelabel')!
    expect(label.textContent).toMatch(/…$/)
    expect(label.querySelector('title')?.textContent).toBe(
      'first_very_long_argument, second_very_long_argument',
    )
  })
})

it('keys only the marks the diagram draws', () => {
  const legendOf = (container: HTMLElement) =>
    container.querySelector('ul.legend')!.textContent!.replace(/\s+/g, ' ')

  const edited = legendOf(render(FlowView, { workflow }).container)
  expect(edited).toContain('entry')
  expect(edited).toContain('previous_result')
  expect(edited).not.toContain('running')
  expect(edited).not.toContain('multiply')
  expect(edited).not.toContain('for_each')

  const running = legendOf(
    render(FlowView, { workflow, activeStep: 'upscale', doneSteps: ['gen'] })
      .container,
  )
  expect(running).toContain('finished')
  expect(running).toContain('running')
})

it('marks the step a failed run stopped in, and fades the ones it never reached', () => {
  const three = {
    steps: [
      ...workflow.steps,
      {
        name: 'save',
        task: {
          command: 'save',
          arguments: { image: 'previous_result:upscale' },
        },
      },
    ],
  }
  const { container } = render(FlowView, {
    workflow: three,
    doneSteps: ['gen'],
    failedStep: 'upscale',
  })
  expect(nodeFor(container, 'upscale').classList.contains('failed')).toBe(true)
  expect(nodeFor(container, 'save').classList.contains('pending')).toBe(true)
  expect(nodeFor(container, 'gen').classList.contains('pending')).toBe(false)
  const legend = container.querySelector('ul.legend')!.textContent!
  expect(legend).toContain('failed')
  expect(legend).toContain('not run')
  expect(legend).not.toContain('running')
})

it('makes a step a control only when it has somewhere to go', () => {
  const { container } = render(FlowView, {
    workflow,
    onselect: () => {},
    selectable: (name: string) => name === 'gen',
  })
  expect(nodeFor(container, 'gen').getAttribute('role')).toBe('button')
  expect(nodeFor(container, 'upscale').hasAttribute('role')).toBe(false)
})

it('shows a lone plain step as one line, not a canvas', () => {
  const { container } = render(FlowView, {
    workflow: { steps: [workflow.steps[0]] },
    doneSteps: ['gen'],
    nodeTimes: { gen: 5.9 },
  })
  expect(container.querySelector('svg')).toBeNull()
  const line = container.querySelector('.loneline')!
  expect(line.classList.contains('done')).toBe(true)
  expect(line.textContent).toContain('gen')
  expect(line.textContent).toContain('Fake')
})

describe('chained segments', () => {
  // The chained-segments template as a realized run carries it: one step
  const chainWorkflow = {
    variables: { segments: 3, anchor: 'segment_1' },
    steps: [
      {
        name: 'clip',
        pipeline: {
          configuration: { component_type: 'ModularPipeline' },
          chain: {
            segments: 'variable:segments',
            anchor: 'variable:anchor',
            trim_frames: 2,
            crossfade_ms: 80,
          },
        },
      },
    ],
  }
  const segmentsOf = (container: HTMLElement) => [
    ...container.querySelectorAll('g.segment'),
  ]

  it('draws a lone chained step as a box of its segments, not one line', () => {
    const { container } = render(FlowView, { workflow: chainWorkflow })
    expect(container.querySelector('.loneline')).toBeNull()
    const clip = nodeFor(container, 'clip')
    expect(clip.getAttribute('aria-label')).toContain('chain of 3 segments')
    expect(clip.querySelector('.chainsummary')?.textContent).toBe(
      'chain · 3 segments',
    )
    const chips = segmentsOf(container)
    expect(
      chips.map((c) => c.querySelector('.chiplabel')?.textContent),
    ).toEqual(['segment 1', 'segment 2', 'segment 3'])
    // The anchor every later segment is pulled toward is segment 1
    expect(chips[0].querySelector('.anchortag')?.textContent).toBe('anchor')
    expect(chips[1].querySelector('.anchortag')).toBeNull()
    // One seam between each pair, saying what crosses it
    expect(
      [...clip.querySelectorAll('.seamlabel')].map((s) => s.textContent),
    ).toEqual(['↓ last frame · trim 2 · 80ms', '↓ last frame · trim 2 · 80ms'])
  })

  it('colors the segments behind the one a run is on, and that one live', () => {
    const { container } = render(FlowView, {
      workflow: chainWorkflow,
      activeStep: 'clip',
      activeSegments: { clip: 2 },
    })
    const states = segmentsOf(container).map((c) =>
      ['done', 'active', 'failed'].find((s) => c.classList.contains(s)),
    )
    expect(states).toEqual(['done', 'active', undefined])
    expect(segmentsOf(container)[1].getAttribute('aria-label')).toBe(
      'segment 2, active',
    )
  })

  it('marks the segment a failed run stopped in, and every one of a finished step', () => {
    const failed = render(FlowView, {
      workflow: chainWorkflow,
      failedStep: 'clip',
      activeSegments: { clip: 3 },
    }).container
    expect(
      segmentsOf(failed).map((c) => c.classList.contains('failed')),
    ).toEqual([false, false, true])
    const done = render(FlowView, {
      workflow: chainWorkflow,
      doneSteps: ['clip'],
    }).container
    expect(segmentsOf(done).every((c) => c.classList.contains('done'))).toBe(
      true,
    )
  })

  it('says where the length comes from when the count is not in the definition', () => {
    const fromAudio = {
      steps: [
        {
          name: 'clip',
          pipeline: { chain: { match_audio: true } },
        },
      ],
    }
    const { container } = render(FlowView, { workflow: fromAudio })
    const clip = nodeFor(container, 'clip')
    expect(clip.querySelector('.chainsummary')?.textContent).toBe(
      'chain · length from audio',
    )
    expect(segmentsOf(container)).toHaveLength(0)
  })
})
