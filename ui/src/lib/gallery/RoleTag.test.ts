import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, describe, expect, it } from 'vitest'
import RoleTag from './RoleTag.svelte'

afterEach(cleanup)

describe('RoleTag', () => {
  it.each([
    ['final', 'Final'],
    ['intermediate', 'Intermediate'],
    ['input', 'Input'],
  ] as const)('renders %s', (role, text) => {
    render(RoleTag, { role })
    const tag = screen.getByText(text)
    expect(tag.dataset.role).toBe(role)
  })

  it('renders nothing for a null role', () => {
    const { container } = render(RoleTag, { role: null })
    expect(container.textContent?.trim()).toBe('')
  })
})
