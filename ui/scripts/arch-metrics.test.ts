// @vitest-environment node
import { spawnSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
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
  // The engine ratchet's line format: harnest's waiver parses it for both
  it('names a metric that rose and ignores one that fell or is new', () => {
    expect(regressions({ a: 3, b: 1, c: 9 }, { a: 2, b: 2 })).toEqual([
      'a: 2 -> 3',
    ])
  })

  it('compares two measurements without measuring', () => {
    const dir = mkdtempSync(join(tmpdir(), 'uimetrics-'))
    writeFileSync(join(dir, 'cur.json'), JSON.stringify({ a: 3, b: 1 }))
    writeFileSync(join(dir, 'base.json'), JSON.stringify({ a: 2, b: 1, c: 0 }))
    const run = spawnSync(
      'node',
      [
        'scripts/arch-metrics.mjs',
        '--compare',
        join(dir, 'cur.json'),
        join(dir, 'base.json'),
      ],
      { encoding: 'utf8' },
    )
    expect(run.status).toBe(1)
    expect(run.stdout.trim()).toBe('a: 2 -> 3')
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

it('does not measure generated code', () => {
  const files = sourceFiles().map((f) => f.replaceAll('\\', '/'))
  expect(files.some((f) => f.includes('/src/lib/generated/'))).toBe(false)
})

const UI = join(dirname(fileURLToPath(import.meta.url)), '..')
const SCRIPT = join(UI, 'scripts', 'arch-metrics.mjs')

describe('what a branch cannot hide', () => {
  it('counts a complex function an inline directive disables', async () => {
    const dir = mkdtempSync(join(UI, '.metrics-probe-'))
    try {
      const file = join(dir, 'probe.ts')
      const branches = Array.from(
        { length: 12 },
        (_, i) => `  if (n === ${i}) return ${i}`,
      ).join('\n')
      writeFileSync(
        file,
        `// eslint-disable-next-line complexity\nexport function f(n: number): number {\n${branches}\n  return -1\n}\n`,
      )
      const { metrics } = await measure([file], ['asset:'])
      expect(metrics.complex_functions).toBe(1)
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('--compare fails as a tool error, not as a rise, on input it cannot read', () => {
    const run = spawnSync(
      'node',
      [SCRIPT, '--compare', join(tmpdir(), 'no-such-measurement.json'), SCRIPT],
      { encoding: 'utf8' },
    )
    expect(run.status).toBe(2)
    expect(run.stdout.trim()).toBe('')
  })
})
