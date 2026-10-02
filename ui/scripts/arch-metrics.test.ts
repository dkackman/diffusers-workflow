// @vitest-environment node
import { describe, expect, it } from 'vitest'
import {
  cycles,
  measure,
  referencePrefixes,
  regressions,
  sourceFiles,
} from './arch-metrics.mjs'

describe('cycles', () => {
  it('finds a two-module cycle and ignores a chain', () => {
    expect(cycles({ a: ['b'], b: ['a'], c: ['a'] })).toEqual([['a', 'b']])
    expect(cycles({ a: ['b'], b: ['c'], c: [] })).toEqual([])
  })

  it('reports a three-module knot as one component', () => {
    expect(cycles({ a: ['b'], b: ['c'], c: ['a'] })).toEqual([['a', 'b', 'c']])
  })
})

describe('referencePrefixes', () => {
  it('reads the prefix constants and nothing else', () => {
    const text =
      'ASSET = "asset:"\nFROM_FILE_KEY = "from_file"\nITEM = "item:"\n'
    expect(referencePrefixes(text)).toEqual(['asset:', 'item:'])
  })

  it('refuses a file with no prefixes rather than measuring 0', () => {
    expect(() => referencePrefixes('nothing here')).toThrow()
  })
})

describe('regressions', () => {
  it('names a metric that rose and ignores one that fell or is new', () => {
    expect(regressions({ a: 3, b: 1, c: 9 }, { a: 2, b: 2 })).toEqual([
      'a: 3 > baseline 2',
    ])
  })
})

describe('the real tree', () => {
  it('excludes tests and measures every metric', async () => {
    const files = sourceFiles()
    expect(files.some((f) => f.endsWith('.test.ts'))).toBe(false)
    expect(files.some((f) => f.endsWith('App.svelte'))).toBe(true)
    const { metrics } = await measure()
    expect(Object.keys(metrics).sort()).toEqual([
      'a11y_suppressions',
      'complex_functions',
      'files_over_size_ceiling',
      'import_cycles',
      'long_functions',
      'modules_in_import_cycles',
      'prefix_literals',
    ])
    for (const value of Object.values(metrics))
      expect(Number.isInteger(value)).toBe(true)
  }, 60_000)
})
