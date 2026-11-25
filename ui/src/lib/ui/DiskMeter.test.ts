import { cleanup, render } from '@testing-library/svelte'
import { afterEach, expect, it } from 'vitest'
import DiskMeter from './DiskMeter.svelte'

const GB = 1024 ** 3

afterEach(cleanup)

it('reads "x / y GB free" with a bar when the total is known', () => {
  const { container } = render(DiskMeter, {
    props: { freeBytes: 400 * GB, totalBytes: 1000 * GB },
  })
  expect(container.textContent?.replace(/\s+/g, ' ').trim()).toBe(
    '400.0 / 1000.0 GB free',
  )
  const bar = container.querySelector('.diskbar > span') as HTMLElement
  expect(bar.getAttribute('style')).toContain('width: 60%')
})

it('reads "x GB free" and has no bar when the total is not known', () => {
  const { container } = render(DiskMeter, { props: { freeBytes: 20 * GB } })
  expect(container.textContent?.replace(/\s+/g, ' ').trim()).toBe(
    '20.0 GB free',
  )
  expect(container.querySelector('.diskbar')).toBeNull()
})
