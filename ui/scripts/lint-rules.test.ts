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
