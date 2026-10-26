import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, describe, expect, it } from 'vitest'
import Skeleton from './Skeleton.svelte'

describe('Skeleton', () => {
  afterEach(cleanup)
  it('announces loading and draws the requested shape', () => {
    render(Skeleton, { props: { tiles: 4, label: 'loading outputs' } })
    const status = screen.getByRole('status', { name: 'loading outputs' })
    expect(status.querySelectorAll('.tile')).toHaveLength(4)
  })
  it('draws lines by default', () => {
    render(Skeleton)
    expect(screen.getByRole('status').querySelectorAll('.line')).toHaveLength(3)
  })
})
