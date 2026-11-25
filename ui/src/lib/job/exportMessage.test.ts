import { expect, it } from 'vitest'
import { exportMessage } from './exportMessage'
import type { JobExport } from '../types'

const result = (files: number, missing: number) =>
  ({
    files: Array(files).fill('f'),
    missing: Array(missing).fill('m'),
    total_bytes: 3 * 1024 * 1024 + 512 * 1024,
    directory: 'exports/abc',
    zip_url: '/zip',
  }) as unknown as JobExport

it('counts the files and the megabytes, and names the directory', () => {
  expect(exportMessage(result(1, 0))).toBe(
    'Exported 1 file (3.5 MB) to exports/abc',
  )
  expect(exportMessage(result(4, 0))).toBe(
    'Exported 4 files (3.5 MB) to exports/abc',
  )
})

it('a small export is not 0.0 MB', () => {
  expect(exportMessage({ ...result(1, 0), total_bytes: 20 * 1024 })).toMatch(
    /\(20 KB\)/,
  )
})

it('says how many referenced files are gone', () => {
  expect(exportMessage(result(2, 1))).toBe(
    'Exported 2 files (3.5 MB) to exports/abc - 1 referenced file no longer on disk',
  )
  expect(exportMessage(result(2, 3))).toMatch(
    / - 3 referenced files no longer on disk$/,
  )
})
