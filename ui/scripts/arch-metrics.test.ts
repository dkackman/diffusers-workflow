// @vitest-environment node
import { spawnSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { api } from '../src/lib/api'
import {
  cycles,
  isLoadState,
  measure,
  misclassified,
  RAW_MEDIA_OWNERS,
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
      'banned_terms',
      'complex_functions',
      'duplicate_blocks',
      'empty_outside_loadstate',
      'files_over_size_ceiling',
      'import_cycles',
      'long_functions',
      'modules_in_import_cycles',
      'prefix_literals',
      'raw_error_toasts',
      'raw_media',
      'unguarded_mutations',
      'unwatched_reads',
      'window_dialogs',
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

describe('the page-quality metrics', () => {
  async function measured(files: Record<string, string>) {
    const dir = mkdtempSync(join(UI, '.metrics-probe-'))
    try {
      const paths = Object.entries(files).map(([name, text]) => {
        const file = join(dir, name)
        writeFileSync(file, text)
        return file
      })
      return (await measure(paths, ['asset:'])).metrics
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  }

  it('counts a raw error toast and not a described one', async () => {
    const m = await measured({
      'a.ts':
        `declare const notify: any\ndeclare const e: any\n` +
        `notify.error(e.message)\nnotify.error(String(e))\n` +
        `notify.failure('save', e)\nnotify.error('plain')\n`,
    })
    expect(m.raw_error_toasts).toBe(2)
  })

  it('counts a raw error passed through a const', async () => {
    const m = await measured({
      'a.svelte': `<script lang="ts">
  declare const notify: any
  function a(e: unknown) {
    const msg = e instanceof Error ? e.message : String(e)
    notify.error(msg)
  }
  function b(e: unknown) {
    let m = String(e)
    notify.error(m)
  }
  function c(e: unknown) {
    let m = 'x'
    m = String(e)
    notify.error(m)
  }
  function d() {
    const ok = 'fine'
    notify.error(ok)
    notify.error('failed: ' + ok)
  }
</script>
`,
    })
    expect(m.raw_error_toasts).toBe(2)
  })

  it('counts an Empty import outside LoadState only', async () => {
    const m = await measured({
      'a.svelte': `<script lang="ts">\n  import Empty from '../ui/Empty.svelte'\n</script>\n<Empty />\n`,
      'LoadState.svelte': `<script lang="ts">\n  import Empty from './Empty.svelte'\n</script>\n<Empty />\n`,
    })
    // LoadState is matched by its path, which a probe dir cannot imitate
    expect(m.empty_outside_loadstate).toBe(2)
    expect(isLoadState(join(UI, 'src/lib/ui/LoadState.svelte'))).toBe(true)
    expect(isLoadState(join(UI, 'src/lib/ui/Empty.svelte'))).toBe(false)
  })

  it('counts a read outside resource and poll, not inside', async () => {
    const svelte = `<script lang="ts">
  import { api } from '../api'
  async function go() {
    await api.listJobs()
  }
  const r = resource(() => api.listJobs(), 'jobs')
  const p = poll(() => api.health(), 5000)
  const q = api.health()
  const w = api.saveWorkflow({})
</script>
`
    const m = await measured({
      'a.svelte': svelte,
      'b.ts': `declare const api: any\napi.listJobs()\n`,
    })
    expect(m.unwatched_reads).toBe(2)
  })
})

describe('the consistency metrics', () => {
  async function measured(files: Record<string, string>, extra: string[] = []) {
    const dir = mkdtempSync(join(UI, '.metrics-probe-'))
    try {
      const paths = Object.entries(files).map(([name, text]) => {
        const file = join(dir, name)
        writeFileSync(file, text)
        return file
      })
      return await measure([...paths, ...extra], ['asset:'])
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  }

  it('window_dialogs counts window.prompt and a bare confirm, not confirmDialog', async () => {
    const { metrics, detail } = await measured({
      'a.svelte': `<script lang="ts">
  declare const confirmDialog: any
  function go() {
    const name = window.prompt('Name')
    if (confirm('Sure?')) confirmDialog({ title: 'Sure?' })
    return name
  }
</script>
<button onclick={go}>Go</button>
`,
      'b.ts': `declare const ui: any\nui.confirm('x')\nexport {}\n`,
    })
    expect(metrics.window_dialogs).toBe(2)
    expect(detail.window_dialogs).toEqual([
      expect.stringMatching(/a\.svelte:4$/),
      expect.stringMatching(/a\.svelte:5$/),
    ])
  })

  it('unguarded_mutations spares a write inside action() and in a function action() names', async () => {
    const { metrics, detail } = await measured({
      'a.svelte': `<script lang="ts">
  import { api } from '../api'
  declare const action: any
  async function helper() {
    await api.cancelJob('j')
  }
  const cancel = action(() => helper())
  const save = action(() => api.saveWorkflow({}))
  async function bare() {
    await api.deleteOutput('x')
  }
  const r = api.listJobs()
</script>
<button onclick={() => api.setStarred('x', true)}>Star</button>
<button onclick={bare}>Delete</button>
`,
      'b.ts': `declare const api: any\napi.deleteOutput('x')\n`,
    })
    expect(metrics.unguarded_mutations).toBe(2)
    expect(detail.unguarded_mutations).toEqual([
      expect.stringMatching(/a\.svelte:10 deleteOutput$/),
      expect.stringMatching(/a\.svelte:14 setStarred$/),
    ])
  })

  it('unguarded_mutations counts a helper an action names when a button also binds it bare', async () => {
    const { metrics, detail } = await measured({
      'a.svelte': `<script lang="ts">
  import { api } from '../api'
  declare const action: any
  async function bare() {
    await api.deleteOutput('x')
  }
  const remove = action(() => bare())
</script>
<button onclick={bare}>Delete</button>
<button onclick={remove.run}>Delete</button>
`,
    })
    expect(metrics.unguarded_mutations).toBe(1)
    expect(detail.unguarded_mutations).toEqual([
      expect.stringMatching(/a\.svelte:5 deleteOutput$/),
    ])
  })

  it('unguarded_mutations: a property or a shadowing local named like the helper does not guard it', async () => {
    const { metrics, detail } = await measured({
      'a.svelte': `<script lang="ts">
  import { api } from '../api'
  declare const action: any
  declare const x: any
  async function helper() {
    await api.cancelJob('j')
  }
  async function other() {
    await api.deleteOutput('x')
  }
  const a = action(() => x.helper())
  const b = action(({ other }: any) => other())
  const c = action(() => ({ helper: 1 }))
</script>
<button onclick={helper}>Cancel</button>
<button onclick={other}>Delete</button>
`,
    })
    expect(metrics.unguarded_mutations).toBe(2)
    expect(detail.unguarded_mutations).toEqual([
      expect.stringMatching(/a\.svelte:6 cancelJob$/),
      expect.stringMatching(/a\.svelte:9 deleteOutput$/),
    ])
  })

  it('unguarded_mutations counts a use of an imported .ts helper that writes, outside action()', async () => {
    const { metrics, detail } = await measured({
      'helpers.ts': `import { api } from './api'
export async function removeIt(name: string) {
  await inner(name)
}
async function inner(name: string) {
  await api.deleteOutput(name)
}
export function label(name: string) {
  return name
}
export { inner as removeInner }
`,
      'a.svelte': `<script lang="ts">
  import { removeIt, label, removeInner } from './helpers'
  import * as h from './helpers'
  declare const action: any
  const guarded = action(() => removeIt('x'))
  const passed = action(removeInner, { doing: 'remove' })
</script>
<button onclick={() => removeIt('y')}>{label('y')}</button>
<button onclick={removeInner}>Remove</button>
<button onclick={() => h.removeIt('z')}>{h.label('z')}</button>
`,
    })
    expect(metrics.unguarded_mutations).toBe(3)
    expect(detail.unguarded_mutations).toEqual([
      expect.stringMatching(/a\.svelte:8 removeIt$/),
      expect.stringMatching(/a\.svelte:9 removeInner$/),
      expect.stringMatching(/a\.svelte:10 removeIt$/),
    ])
  })

  it('banned_terms skips style text and terms.ts', async () => {
    const { metrics, detail } = await measured(
      {
        'a.svelte': `<script lang="ts">
  const eta = \`~\${3} min\`
  const word = 'session'
  const hint = 'No session is listening'
</script>
<p>Keep as asset</p>
<span title="held for your review">x</span>
<style>
  /* the run note, ~3 */
  p { margin: 0 }
</style>
`,
      },
      [join(UI, 'src/lib/terms.ts')],
    )
    expect(detail.banned_terms).toEqual([
      expect.stringMatching(/a\.svelte:2 tilde$/),
      expect.stringMatching(/a\.svelte:4 session$/),
      expect.stringMatching(/a\.svelte:6 keep-asset$/),
      expect.stringMatching(/a\.svelte:7 held-review$/),
    ])
    expect(metrics.banned_terms).toBe(4)
  })

  it('raw_media counts img, video and audio outside OutputTile and MediaBody', async () => {
    const { metrics, detail } = await measured({
      'a.svelte': `<img src="a.png" alt="" />
{#if x}
  <video src="b.mp4"></video>
{/if}
<audio src="c.mp3" controls></audio>
<picture></picture>
`,
    })
    expect(metrics.raw_media).toBe(3)
    expect(detail.raw_media).toEqual([
      expect.stringMatching(/a\.svelte:1 img$/),
      expect.stringMatching(/a\.svelte:3 video$/),
      expect.stringMatching(/a\.svelte:5 audio$/),
    ])
  })

  it('raw_media spares exactly OutputTile and MediaBody', () => {
    expect([...RAW_MEDIA_OWNERS].sort()).toEqual([
      'src/lib/ui/MediaBody.svelte',
      'src/lib/ui/OutputTile.svelte',
    ])
  })

  it('duplicate_blocks counts one shared 6-line run once per file', async () => {
    const block = Array.from(
      { length: 8 },
      (_, i) => `  <p class="row-${i}">line ${i}</p>`,
    ).join('\n')
    const { metrics } = await measured({
      'a.svelte': `<div>\n${block}\n</div>\n<p>only a</p>\n`,
      'b.svelte': `<section>\n  <h2>b</h2>\n${block.replaceAll('  <p', '      <p')}\n</section>\n`,
      'c.svelte': `<p>nothing shared</p>\n`,
    })
    expect(metrics.duplicate_blocks).toBe(2)
  })
})

describe('the read classification', () => {
  // A new api method fails here until someone says whether it is a read, so
  // unwatched_reads cannot miss it by its name
  it('puts every api key in exactly one of the reads and NOT_READS', () => {
    expect(misclassified(Object.keys(api))).toEqual([])
  })

  it('names a key in neither list, one in both, and a stale entry', () => {
    expect(
      misclassified(
        ['listThings', 'fetchThing', 'saveThing', 'thingAlias', 'getThing'],
        ['thingAlias', 'gone'],
        ['saveThing', 'getThing'],
      ),
    ).toEqual([
      'fetchThing: neither a read nor in NOT_READS',
      'getThing: both a read and in NOT_READS',
      'gone: listed, but api has no such key',
    ])
  })
})
