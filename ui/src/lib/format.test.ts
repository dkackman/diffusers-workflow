import { expect, it } from 'vitest'
import { formatBytes, formatMtime } from './format'

it('formats sub-megabyte sizes in whole KB, rounded up to at least 1', () => {
  expect(formatBytes(500)).toBe('1 KB')
  expect(formatBytes(2048)).toBe('2 KB')
  expect(formatBytes(1024 * 1023)).toBe('1023 KB')
})

it('formats megabyte sizes to one decimal at the 1 MB boundary', () => {
  expect(formatBytes(1024 * 1024)).toBe('1.0 MB')
  expect(formatBytes(1024 * 1024 * 5)).toBe('5.0 MB')
})

it('formats gigabyte sizes to two decimals at and above 1 GB', () => {
  expect(formatBytes(1024 ** 3)).toBe('1.00 GB')
  expect(formatBytes(1024 ** 3 * 2.5)).toBe('2.50 GB')
})

it('reads mtime as unix seconds, not milliseconds', () => {
  // 1700000000 s is 2023-11-14T22:13:20Z; read as ms it would land in January 1970
  const rendered = formatMtime(1700000000)
  expect(rendered).toBe(new Date('2023-11-14T22:13:20Z').toLocaleString())
  expect(rendered).not.toBe(new Date(1700000000).toLocaleString())
})
