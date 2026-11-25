// @vitest-environment node
// The change feed's one-subscriber rule (eslint.config.js), run on code that
// tries each way round it: every spelling of a subscription outside
// resource() fails lint, and the ChangeKind type stays importable
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { ESLint } from 'eslint'
import { expect, it } from 'vitest'

const UI = join(dirname(fileURLToPath(import.meta.url)), '..')
const eslint = new ESLint({ cwd: UI })

/** The messages of the feed rule `code` breaks, linted as `file` */
async function feedErrors(code: string, file: string): Promise<string[]> {
  const [result] = await eslint.lintText(code, { filePath: join(UI, file) })
  return result.messages
    .filter((m) => m.message.includes('Subscribe through resource()'))
    .map((m) => m.ruleId ?? '')
}

const SPELLINGS: Record<string, string> = {
  chained: `import { changes } from './changes.svelte'
changes('w').watch(['jobs'], () => {})`,
  held: `import { changes } from './changes.svelte'
const feed = changes('w')
feed.watch(['jobs'], () => {})`,
  constructed: `import { ChangeFeed } from './changes.svelte'
new ChangeFeed('w').watch(['jobs'], () => {})`,
  aliased: `import { changes as feedOf } from '../changes.svelte'
feedOf('w').watch(['jobs'], () => {})`,
  namespace: `import * as feeds from './changes.svelte'
feeds.changes('w').watch(['jobs'], () => {})`,
  dynamic: `const feeds = await import('./changes.svelte')
const feed = feeds.changes('w')
feed.watch(['jobs'], () => {})`,
}

it.each(Object.entries(SPELLINGS))(
  'a subscription outside resource(), %s, fails lint',
  async (_, code) => {
    expect(await feedErrors(code, 'src/lib/pages/Probe.ts')).not.toEqual([])
    expect(await feedErrors(code, 'src/lib/ui/Probe.ts')).not.toEqual([])
  },
)

it('the ChangeKind type stays importable anywhere', async () => {
  const code = `import type { ChangeKind } from './changes.svelte'
export const k: ChangeKind = 'jobs'`
  expect(await feedErrors(code, 'src/lib/pages/Probe.ts')).toEqual([])
})

it('resource() itself subscribes', async () => {
  expect(
    await feedErrors(SPELLINGS.held, 'src/lib/resource.svelte.ts'),
  ).toEqual([])
})

// The raw-message rule (eslint.config.js): a caught error's .message shown
// for display fails lint everywhere but describeError's own module and tests
/** The messages of the raw-message rule `code` breaks, linted as `file` */
async function messageErrors(code: string, file: string): Promise<string[]> {
  const [result] = await eslint.lintText(code, { filePath: join(UI, file) })
  return result.messages
    .filter((m) => m.message.includes('never its raw .message'))
    .map((m) => m.ruleId ?? '')
}

// The ternary alone: `String(e)` is a form of its own (RAW_FORMS)
const RAW = `export function said(e: unknown) {
  return e instanceof Error ? e.message : 'failed'
}`

it.each([
  'src/lib/pages/Probe.ts',
  'src/lib/ui/Probe.ts',
  'src/lib/changes.svelte.ts',
])('reading a caught error for display fails lint in %s', async (file) => {
  expect(await messageErrors(RAW, file)).toEqual(['no-restricted-syntax'])
})

it('the feed rule still holds beside the raw-message rule', async () => {
  const both = `${RAW}\n${SPELLINGS.chained}`
  const [result] = await eslint.lintText(both, {
    filePath: join(UI, 'src/lib/pages/Probe.ts'),
  })
  const ids = result.messages.map((m) => m.message)
  expect(ids.some((m) => m.includes('never its raw .message'))).toBe(true)
  expect(ids.some((m) => m.includes('Subscribe through resource()'))).toBe(true)
})

it.each(['src/lib/apiErrors.ts', 'src/lib/pages/Probe.test.ts'])(
  'describeError itself and the tests may read it: %s',
  async (file) => {
    expect(await messageErrors(RAW, file)).toEqual([])
  },
)

it('a component reading a caught error for display fails lint', async () => {
  const code = `<script lang="ts">\n  let shown = $state('')\n  try {\n    JSON.parse('{')\n  } catch (e) {\n    shown = e instanceof Error ? e.message : 'failed'\n  }\n</script>\n\n<p>{shown}</p>\n`
  expect(await messageErrors(code, 'src/lib/pages/Probe.svelte')).toEqual([
    'no-restricted-syntax',
  ])
})

// The delete rule (eslint.config.js): a confirm whose button deletes or
// removes is built by destroyConfirm, so every delete asks one way
/** The messages of the delete rule `code` breaks, linted as `file` */
async function destroyErrors(code: string, file: string): Promise<string[]> {
  const [result] = await eslint.lintText(code, { filePath: join(UI, file) })
  return result.messages
    .filter((m) => m.message.includes('asks through destroyConfirm()'))
    .map((m) => m.ruleId ?? '')
}

const ASKS = `import { confirmDialog } from './confirm.svelte'
export const ask = () => confirmDialog('Gone?', { confirmLabel: 'Delete' })
export const drop = () => confirmDialog('Gone?', { confirmLabel: 'Remove it' })`

it('a Delete confirm outside destructive.ts fails lint', async () => {
  for (const file of [
    'src/lib/pages/Probe.ts',
    'src/lib/ui/Probe.ts',
    'src/lib/resource.svelte.ts',
    'src/lib/apiErrors.ts',
  ])
    expect(await destroyErrors(ASKS, file)).toEqual([
      'no-restricted-syntax',
      'no-restricted-syntax',
    ])
  const svelte = `<script lang="ts">\n  import { confirmDialog } from '../confirm.svelte'\n  const ask = () => confirmDialog('Gone?', { confirmLabel: 'Delete' })\n</script>\n\n<button onclick={ask}>x</button>\n`
  expect(await destroyErrors(svelte, 'src/lib/pages/Probe.svelte')).toEqual([
    'no-restricted-syntax',
  ])
  expect(await destroyErrors(ASKS, 'src/lib/destructive.ts')).toEqual([])
  // Other labels are not deletes
  const keep = `export const spec = { confirmLabel: 'Replace' }`
  expect(await destroyErrors(keep, 'src/lib/pages/Probe.ts')).toEqual([])
})

it('destructive.ts keeps the other syntax rules', async () => {
  expect(await messageErrors(RAW, 'src/lib/destructive.ts')).toEqual([
    'no-restricted-syntax',
  ])
})

// The Monaco rule (eslint.config.js): the whole package - every language
// it ships - is imported nowhere; Monaco loads through src/lib/monaco.ts,
// and a type import costs nothing at run time
/** The messages of the Monaco rule `code` breaks, linted as `file` */
async function monacoErrors(code: string, file: string): Promise<string[]> {
  const [result] = await eslint.lintText(code, { filePath: join(UI, file) })
  return result.messages
    .filter((m) =>
      m.message.includes(
        'Import Monaco through src/lib/monaco.ts, which loads the JSON language only.',
      ),
    )
    .map((m) => m.ruleId ?? '')
}

it('importing monaco-editor whole is refused', async () => {
  const whole = `import * as monaco from 'monaco-editor'\nexport const m = monaco`
  for (const file of [
    'src/lib/monaco.ts',
    'src/lib/pages/Probe.ts',
    'src/lib/ui/Probe.ts',
    'src/lib/resource.svelte.ts',
    'src/lib/pages/Probe.test.ts',
  ])
    expect(await monacoErrors(whole, file)).toEqual(['no-restricted-imports'])
  const svelte = `<script lang="ts">\n  import * as monaco from 'monaco-editor'\n  export const m = monaco\n</script>\n`
  for (const file of ['src/lib/ui/LoadState.svelte', 'src/lib/pages/P.svelte'])
    expect(await monacoErrors(svelte, file)).toEqual(['no-restricted-imports'])
})

it('a type import is allowed', async () => {
  const typed = `import type * as Monaco from 'monaco-editor'\nexport let e: Monaco.editor.IStandaloneCodeEditor | null = null`
  expect(await monacoErrors(typed, 'src/lib/editor/Probe.ts')).toEqual([])
  const core = `import * as monaco from 'monaco-editor/editor/editor.api.js'\nexport const m = monaco`
  expect(await monacoErrors(core, 'src/lib/monaco.ts')).toEqual([])
})

it('a part of monaco-editor is imported only by monaco.ts and monacoContribs.ts', async () => {
  const part = `import 'monaco-editor/basic-languages/python/python.contribution.js'\nimport * as api from 'monaco-editor/editor/editor.api.js'\nexport const a = api`
  for (const file of [
    'src/lib/pages/Probe.ts',
    'src/lib/ui/Probe.ts',
    'src/lib/resource.svelte.ts',
    'src/lib/ui/LoadState.svelte',
    'src/lib/editor/Probe.test.ts',
  ]) {
    const code = file.endsWith('.svelte')
      ? `<script lang="ts">\n  ${part.replaceAll('\n', '\n  ')}\n</script>\n`
      : part
    expect(await monacoErrors(code, file), file).toEqual([
      'no-restricted-imports',
      'no-restricted-imports',
    ])
  }
  for (const file of ['src/lib/monaco.ts', 'src/lib/monacoContribs.ts'])
    expect(await monacoErrors(part, file), file).toEqual([])
  const typed = `import type * as Api from 'monaco-editor/editor/editor.api.js'\nexport let e: Api.editor.IStandaloneCodeEditor | null = null`
  expect(await monacoErrors(typed, 'src/lib/editor/Probe.ts')).toEqual([])
})

// Each other way a caught error reaches the screen raw
const RAW_FORMS: Record<string, string> = {
  'an if guard': `export function said(e: unknown) {
  if (e instanceof Error) return e.message
  return 'failed'
}`,
  'an && guard': `export const said = (e: unknown) => e instanceof Error && e.message`,
  'a cast': `export const said = (e: unknown) => (e as Error).message`,
  'String(e)': `export function said(err: unknown) {
  return String(err)
}`,
  'a template': `export function said(error: unknown) {
  return \`failed: \${error}\`
}`,
}

it.each(Object.entries(RAW_FORMS))(
  'a caught error shown raw through %s fails lint',
  async (_, code) => {
    expect(await messageErrors(code, 'src/lib/pages/Probe.ts')).toEqual([
      'no-restricted-syntax',
    ])
    expect(await messageErrors(code, 'src/lib/apiErrors.ts')).toEqual([])
  },
)

it('a value that is not a caught error is stringified freely', async () => {
  const code = `export const n = (count: number) => String(count) + \`\${count}\``
  expect(await messageErrors(code, 'src/lib/pages/Probe.ts')).toEqual([])
})

// Each other way a delete button is worded
const DELETE_FORMS: Record<string, [string, string]> = {
  'a template confirmLabel': [
    'src/lib/pages/Probe.ts',
    `import { confirmDialog } from './confirm.svelte'
export const ask = (n: number) => confirmDialog('Gone?', { confirmLabel: \`Delete \${n}\` })`,
  ],
  'a chooseDialog label': [
    'src/lib/pages/Probe.ts',
    `import { chooseDialog } from './confirm.svelte'
export const ask = () => chooseDialog('Gone?', [{ label: 'Delete all', value: 'all' }])`,
  ],
  'a templated chooseDialog label': [
    'src/lib/pages/Probe.ts',
    `import { chooseDialog } from './confirm.svelte'
export const ask = (n: number) => chooseDialog('Gone?', [{ label: \`Remove the \${n}\`, value: 'n' }])`,
  ],
  'a markup confirmLabel': [
    'src/lib/pages/Probe.svelte',
    `<script lang="ts">\n  import Ask from './Ask.svelte'\n</script>\n\n<Ask confirmLabel="Delete file" />\n`,
  ],
  'a braced markup confirmLabel': [
    'src/lib/pages/Probe.svelte',
    `<script lang="ts">\n  import Ask from './Ask.svelte'\n  let n = 2\n</script>\n\n<Ask confirmLabel={\`Delete \${n}\`} />\n`,
  ],
}

it.each(Object.entries(DELETE_FORMS))(
  'a delete worded through %s fails lint outside destructive.ts',
  async (_, [file, code]) => {
    expect(await destroyErrors(code, file)).toEqual(['no-restricted-syntax'])
  },
)

it('a chooseDialog label that does not delete, or a label outside one, passes', async () => {
  const code = `import { chooseDialog } from './confirm.svelte'
export const ask = () => chooseDialog('Keep?', [{ label: 'Keep both', value: 'both' }])
export const menu = [{ label: 'Delete', value: 'x' }]`
  expect(await destroyErrors(code, 'src/lib/pages/Probe.ts')).toEqual([])
  expect(
    await destroyErrors(
      DELETE_FORMS['a chooseDialog label'][1],
      'src/lib/destructive.ts',
    ),
  ).toEqual([])
})
