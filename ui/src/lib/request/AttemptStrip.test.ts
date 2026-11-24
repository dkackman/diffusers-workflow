import { cleanup, render } from '@testing-library/svelte'
import { afterEach, expect, it } from 'vitest'
import AttemptStrip from './AttemptStrip.svelte'
import { attempt, item } from './fixtures'

afterEach(cleanup)

const strip = (reason?: string) =>
  render(AttemptStrip, {
    item: item('a', {
      status: 'failed',
      attempts: [attempt('j1', 'failed', reason ? { reason } : {})],
    }),
    workspace: 'studio',
  })

it('a failed tile shows its reason', () => {
  const { container } = strip('CUDA out of memory')
  expect(container.querySelector('.reason')?.textContent).toBe(
    'CUDA out of memory',
  )
})

it('a long reason is one line with the whole in its title', () => {
  const long = 'x'.repeat(300)
  const { container } = strip(long)
  const el = container.querySelector('.reason') as HTMLElement
  expect(el.getAttribute('title')).toBe(long)
  expect(el.classList.contains('bad')).toBe(true)
})

it('a failed take with no reason shows no reason line', () => {
  const { container } = strip()
  expect(container.querySelector('.reason')).toBeNull()
})
