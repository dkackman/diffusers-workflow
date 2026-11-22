// A stand-in for the change feed (changes.svelte.ts) in a component test:
// it records what each view watches, and `move` calls the watchers of a
// kind as a moved counter would. The feed's own timing is changes.test.ts's.
import type { ChangeKind } from '../changes.svelte'

export interface Watch {
  workspace: string
  kinds: readonly ChangeKind[]
  fn: () => unknown
}

export function feedMock() {
  const watches = new Set<Watch>()
  return {
    watches,
    changes: (workspace: string) => ({
      watch(kinds: readonly ChangeKind[], fn: () => unknown) {
        const entry = { workspace, kinds, fn }
        watches.add(entry)
        return () => watches.delete(entry)
      },
    }),
    /** What is watched in `workspace`, by kind */
    watched: (workspace: string) =>
      [...watches]
        .filter((w) => w.workspace === workspace)
        .flatMap((w) => w.kinds),
    /** `kind` moved in `workspace`: call its watchers */
    move(workspace: string, kind: ChangeKind) {
      for (const w of [...watches])
        if (w.workspace === workspace && w.kinds.includes(kind)) void w.fn()
    },
  }
}

/** The one a test file mocks the feed with:
 *   vi.mock('../changes.svelte', async () => ({
 *     changes: (await import('../test/feedMock')).feed.changes,
 *   }))
 * and imports itself to `move` a kind */
export const feed = feedMock()
