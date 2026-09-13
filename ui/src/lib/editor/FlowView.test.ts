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
