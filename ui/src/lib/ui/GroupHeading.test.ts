import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import GroupHeading from './GroupHeading.svelte'

afterEach(cleanup)

it('tells screen readers whether it is open', async () => {
  const ontoggle = vi.fn()
  const { rerender } = render(GroupHeading, {
    props: { label: 'shared/', count: 3, collapsed: false, ontoggle },
  })
  const button = screen.getByRole('button', { name: /shared\/ \(3\)/ })
  expect(button.getAttribute('aria-expanded')).toBe('true')
  await fireEvent.click(button)
  expect(ontoggle).toHaveBeenCalledOnce()
  await rerender({ label: 'shared/', count: 3, collapsed: true, ontoggle })
  expect(button.getAttribute('aria-expanded')).toBe('false')
})
