import { render } from '@testing-library/svelte'
import { expect, it } from 'vitest'
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
