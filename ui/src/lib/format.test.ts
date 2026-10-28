import { describe, expect, it } from 'vitest'
import {
  formatBytes,
  formatDuration,
  formatMinutes,
  formatDay,
  formatMtime,
  gbFromBytes,
  gbFromMb,
} from './format'

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
  const now = new Date('2023-12-01T00:00:00Z')
  const rendered = formatMtime(1700000000, now)
  expect(rendered).toBe(
    formatMtime(Date.parse('2023-11-14T22:13:20Z') / 1000, now),
  )
  expect(rendered).not.toBe(formatMtime(1700000, now))
})

it('always carries the date, and the year only when it is not this one', () => {
  const stamp = Date.parse('2023-11-14T22:13:20Z') / 1000
  const thisYear = formatMtime(stamp, new Date('2023-12-01T00:00:00Z'))
  const lastYear = formatMtime(stamp, new Date('2024-02-01T00:00:00Z'))
  expect(thisYear).toMatch(/Nov/)
  expect(thisYear).not.toMatch(/2023/)
  expect(lastYear).toMatch(/2023/)
})

describe('GB figures', () => {
  it('reads megabytes and bytes alike, one decimal, no unit', () => {
    expect(gbFromMb(1536)).toBe('1.5')
    expect(gbFromBytes(1610612736)).toBe('1.5')
    expect(gbFromMb(0)).toBe('0.0')
  })
})

describe('formatDuration', () => {
  it('reads seconds, then minutes, then hours', () => {
    expect(formatDuration(0)).toBe('0.0s')
    expect(formatDuration(12.34)).toBe('12.3s')
    expect(formatDuration(243)).toBe('4m 03s')
    expect(formatDuration(3725)).toBe('1h 02m')
  })
})

describe('formatMinutes', () => {
  it('rounds to two places and says <1 below a minute', () => {
    expect(formatMinutes(12.946)).toBe('~12.95 min')
    expect(formatMinutes(0.4)).toBe('<1 min')
  })
})

it('gives a day without its time, and the year only from another one', () => {
  const stamp = Date.parse('2023-11-14T22:13:20Z') / 1000
  const thisYear = formatDay(stamp, new Date('2023-12-01T00:00:00Z'))
  expect(thisYear).toMatch(/Nov/)
  expect(thisYear).not.toMatch(/2023|:/)
  expect(formatDay(stamp, new Date('2024-02-01T00:00:00Z'))).toMatch(/2023/)
})
