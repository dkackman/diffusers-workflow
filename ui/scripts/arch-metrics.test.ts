// @vitest-environment node
import { spawnSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { api } from '../src/lib/api'
import * as apiModule from '../src/lib/api'
import {
  cycles,
  EXPORTED_NOT_WRITES,
  EXPORTED_WRITES,
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
  import { action } from '../action.svelte'
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
  import { action } from '../action.svelte'
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
  import { action } from '../action.svelte'
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
  import { action } from '../action.svelte'
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
      expect.stringMatching(/a\.svelte:3 session$/),
      expect.stringMatching(/a\.svelte:5 keep-asset$/),
      expect.stringMatching(/a\.svelte:6 held-review$/),
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
`,
    })
    expect(metrics.raw_media).toBe(3)
    expect(detail.raw_media).toEqual([
      expect.stringMatching(/a\.svelte:1 img$/),
      expect.stringMatching(/a\.svelte:3 video$/),
      expect.stringMatching(/a\.svelte:5 audio$/),
    ])
  })

  it('raw_media counts source and picture, a nested one with its holder', async () => {
    const { metrics, detail } = await measured({
      'a.svelte': `<picture><source srcset="a.webp" /><img src="a.png" alt="" /></picture>
<video><source src="b.mp4" /><track kind="captions" /></video>
<source src="c.mp3" />
`,
    })
    expect(detail.raw_media).toEqual([
      expect.stringMatching(/a\.svelte:1 picture$/),
      expect.stringMatching(/a\.svelte:2 video$/),
      expect.stringMatching(/a\.svelte:3 source$/),
    ])
    expect(metrics.raw_media).toBe(3)
  })

  it('raw_media counts svelte:element whose this is, or can be, a media tag', async () => {
    const { metrics, detail } = await measured({
      'a.svelte': `<svelte:element this="img" src="a.png" />
<svelte:element this={'video'} />
<svelte:element this="div" />
<svelte:element this={tag} />
<svelte:element this={big ? 'h3' : 'audio'} />
<svelte:element this={nested ? 'h4' : 'h3'} />
<svelte:element this={kind || 'canvas'} />
`,
    })
    expect(detail.raw_media).toEqual([
      expect.stringMatching(/a\.svelte:1 img$/),
      expect.stringMatching(/a\.svelte:2 video$/),
      expect.stringMatching(/a\.svelte:5 audio$/),
      expect.stringMatching(/a\.svelte:7 canvas$/),
    ])
    expect(metrics.raw_media).toBe(4)
  })

  it('raw_media counts media made in script, in .ts and .svelte', async () => {
    const { metrics, detail } = await measured({
      'a.ts': `export const a = document.createElement('video')
export const b = new Audio('x.mp3')
export const c = new Image()
export const d = document.createElement(\`audio\`)
export const ok = document.createElement('a')
`,
      'b.svelte': `<script lang="ts">
  const el = document.createElement('img')
  const t = document.createElement('textarea')
</script>
`,
    })
    expect(metrics.raw_media).toBe(5)
    expect(detail.raw_media).toEqual(
      expect.arrayContaining([
        expect.stringMatching(/a\.ts:1 video$/),
        expect.stringMatching(/a\.ts:2 new Audio$/),
        expect.stringMatching(/a\.ts:3 new Image$/),
        expect.stringMatching(/a\.ts:4 audio$/),
        expect.stringMatching(/b\.svelte:2 img$/),
      ]),
    )
  })

  it('raw_media counts {@html} holding a media tag', async () => {
    const { metrics } = await measured({
      'a.svelte': `{@html '<img src="a.png">'}
{@html \`<video src="\${u}"></video>\`}
{@html '<b>bold</b>'}
`,
    })
    expect(metrics.raw_media).toBe(2)
  })

  it('raw_media counts track, embed, object and canvas', async () => {
    const { metrics, detail } = await measured({
      'a.svelte': `<track src="a.vtt" />
<embed src="b.pdf" />
<object data="c.svg" title="c"></object>
<canvas></canvas>
`,
    })
    expect(detail.raw_media).toEqual([
      expect.stringMatching(/a\.svelte:1 track$/),
      expect.stringMatching(/a\.svelte:2 embed$/),
      expect.stringMatching(/a\.svelte:3 object$/),
      expect.stringMatching(/a\.svelte:4 canvas$/),
    ])
    expect(metrics.raw_media).toBe(4)
  })

  it('raw_media follows {@html} through a variable of the same file', async () => {
    const { metrics, detail } = await measured({
      'a.svelte': `<script lang="ts">
  const picture = '<img src="a.png">'
  let later = ''
  later = \`<canvas></canvas>\`
  const words = '<b>bold</b>'
</script>
{@html picture}
{@html later}
{@html words}
`,
    })
    expect(detail.raw_media).toEqual([
      expect.stringMatching(/a\.svelte:7 html$/),
      expect.stringMatching(/a\.svelte:8 html$/),
    ])
    expect(metrics.raw_media).toBe(2)
  })

  it('raw_media counts a CSS background picture, in a stylesheet and a component', async () => {
    const { metrics, detail } = await measured({
      'a.css': `.hero {
  background-image: url('a.png');
}
.plain { background: var(--bg); }
.short { background: #000 url(b.png) no-repeat; }
`,
      'b.svelte': `<div class="x" style="background-image: url({src})"></div>
<style>
  .x { background-color: red; }
  .y { background-image:url(c.png) }
</style>
`,
    })
    expect(detail.raw_media).toEqual([
      expect.stringMatching(/a\.css:2 css background$/),
      expect.stringMatching(/a\.css:5 css background$/),
      expect.stringMatching(/b\.svelte:1 css background$/),
      expect.stringMatching(/b\.svelte:4 css background$/),
    ])
    expect(metrics.raw_media).toBe(4)
  })

  it('window_dialogs counts globalThis, self and a computed name', async () => {
    const { metrics, detail } = await measured({
      'a.ts': `globalThis.confirm('a')
self.alert('b')
window['confirm']('c')
globalThis['prompt']('d')
window[name]('e')
other.alert('f')
export {}
`,
    })
    expect(detail.window_dialogs).toEqual([
      expect.stringMatching(/a\.ts:1$/),
      expect.stringMatching(/a\.ts:2$/),
      expect.stringMatching(/a\.ts:3$/),
      expect.stringMatching(/a\.ts:4$/),
    ])
    expect(metrics.window_dialogs).toBe(4)
  })

  it('unguarded_mutations: only the action action.svelte exports guards, under any local name', async () => {
    const { metrics, detail } = await measured({
      'a.svelte': `<script lang="ts">
  import { api } from '../api'
  import { action as act } from '../action.svelte'
  const ok = act(() => api.cancelJob('j'))
</script>
`,
      'b.svelte': `<script lang="ts">
  import { api } from '../api'
  import { action } from './elsewhere'
  const notOk = action(() => api.cancelJob('j'))
</script>
`,
      'c.svelte': `<script lang="ts">
  import { api } from '../api'
  const action = (f: () => unknown) => f
  const notOk = action(() => api.deleteOutput('x'))
</script>
`,
    })
    expect(detail.unguarded_mutations).toEqual([
      expect.stringMatching(/b\.svelte:4 cancelJob$/),
      expect.stringMatching(/c\.svelte:4 deleteOutput$/),
    ])
    expect(metrics.unguarded_mutations).toBe(2)
  })

  it("unguarded_mutations counts a use of api.ts's exported writer setNote", async () => {
    const { metrics, detail } = await measured(
      {
        'a.svelte': `<script lang="ts">
  import { setNote, outputUrl } from '../src/lib/api'
  import { action } from '../src/lib/action.svelte'
  const save = action(() => setNote('t', 'x'), { doing: 'save' })
</script>
<button onclick={() => setNote('t', null)}>Clear</button>
<img alt="" src={outputUrl('x')} />
`,
      },
      [join(UI, 'src/lib/api.ts')],
    )
    expect(detail.unguarded_mutations).toEqual([
      expect.stringMatching(/a\.svelte:6 setNote$/),
    ])
    expect(metrics.unguarded_mutations).toBe(1)
  })

  it('unguarded_mutations follows a write handed to a method that guards it (picks.svelte.ts)', async () => {
    const { metrics, detail } = await measured({
      'picks.svelte.ts': `import { action } from './action.svelte'
export class Picks {
  download(archive: (n: string[]) => Promise<unknown>, noun: string) {
    const a = action((p: string[]) => archive(p), { doing: noun })
    return a
  }
  star(set: (n: string, v: boolean) => Promise<unknown>) {
    const put = async (n: string) => set(n, true)
    return action(async (p: string[]) => put(p[0]), { doing: 'star' })
  }
  leak(write: (n: string) => Promise<unknown>) {
    return write
  }
}
export function guarded(write: () => Promise<unknown>) {
  return action(() => write(), { doing: 'x' })
}
`,
      'a.svelte': `<script lang="ts">
  import { api } from '../api'
  import { action } from '../action.svelte'
  import { Picks, guarded } from './picks.svelte'
  const picks = new Picks()
  const d = picks.download(api.archiveOutputs, 'files')
  const s = picks.star(api.setStarred)
  const g = guarded(api.clearCache)
  const w = action(api.deleteAsset, { doing: 'delete' })
  const l = picks.leak(api.deleteOutput)
  const bare = api.cancelJob
  const read = picks.leak(api.listJobs)
</script>
`,
    })
    expect(detail.unguarded_mutations).toEqual([
      expect.stringMatching(/a\.svelte:10 deleteOutput$/),
      expect.stringMatching(/a\.svelte:11 cancelJob$/),
    ])
    expect(metrics.unguarded_mutations).toBe(2)
  })

  it('banned_terms checks single words, and a literal in a mustache', async () => {
    const { metrics, detail } = await measured({
      'a.svelte': `<script lang="ts">
  import x from './starters'
  const heading = 'Starters'
</script>
<p>{'(the run note)'}</p>
<p>{\`Session\`}</p>
`,
    })
    expect(detail.banned_terms).toEqual([
      expect.stringMatching(/a\.svelte:3 starter$/),
      expect.stringMatching(/a\.svelte:5 run-noun$/),
      expect.stringMatching(/a\.svelte:6 session$/),
    ])
    expect(metrics.banned_terms).toBe(3)
  })

  it('the banned detail is sorted', async () => {
    // lint order: b.svelte before a.svelte, line 10 before line 9, and
    // `tilde` (listed first in BANNED) before `session` on one piece of copy
    const { detail } = await measured({
      'b.svelte': `<p>~3 min</p>\n`,
      'a.svelte': `${'<p>x</p>\n'.repeat(8)}<p>{'~3 session'}</p>\n<p>{'Starters'}</p>\n`,
    })
    const found = detail.banned_terms.map((d: string) =>
      d.replace(/^.*\/(\w\.svelte)/, '$1'),
    )
    expect(found).toEqual([
      'a.svelte:9 session',
      'a.svelte:9 tilde',
      'a.svelte:10 starter',
      'b.svelte:1 tilde',
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

describe("api.ts's exports outside api", () => {
  // A new export fails here until someone says whether it writes, so
  // unguarded_mutations cannot miss it
  it('puts every export in exactly one of EXPORTED_WRITES and EXPORTED_NOT_WRITES', () => {
    const exported = Object.keys(apiModule).filter((k) => k !== 'api')
    const listed = [...EXPORTED_WRITES, ...EXPORTED_NOT_WRITES]
    expect([...listed].sort()).toEqual([...exported].sort())
    expect(new Set(listed).size).toBe(listed.length)
  })
})

it('the metrics script holds the complexity limit it measures', async () => {
  const { metrics } = await measure([SCRIPT], ['asset:'])
  expect(metrics.complex_functions).toBe(0)
}, 60_000)

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
