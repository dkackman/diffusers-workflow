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

const RAW = `export function said(e: unknown) {
  return e instanceof Error ? e.message : String(e)
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
  const code = `<script lang="ts">\n  let shown = $state('')\n  try {\n    JSON.parse('{')\n  } catch (e) {\n    shown = e instanceof Error ? e.message : String(e)\n  }\n</script>\n\n<p>{shown}</p>\n`
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
