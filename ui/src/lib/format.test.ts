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

it('renders a unix timestamp as a locale date/time string', () => {
  const mtime = 1700000000
  expect(formatMtime(mtime)).toBe(new Date(mtime * 1000).toLocaleString())
})
