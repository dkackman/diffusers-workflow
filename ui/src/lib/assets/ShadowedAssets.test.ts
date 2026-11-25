import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it } from 'vitest'
import ShadowedAssets from './ShadowedAssets.svelte'
import type { ShadowedAsset } from '../types'

afterEach(cleanup)

const entries = [
  {
    name: 'a.png',
    kind: 'image',
    origin: 'common',
    shadowed_by: 'workspace',
    reference: 'asset:a.png',
  },
] as unknown as ShadowedAsset[]

it('the toggle is disabled while filtering', () => {
  const { rerender } = render(ShadowedAssets, { entries, filterActive: false })
  const toggle = () => screen.getByRole('button', { name: /shadowed/ })
  expect((toggle() as HTMLButtonElement).disabled).toBe(false)
  void rerender({ entries, filterActive: true })
  return Promise.resolve().then(() => {
    expect((toggle() as HTMLButtonElement).disabled).toBe(true)
    expect(toggle().title).toBe('Showing every match while the filter is on')
  })
})
