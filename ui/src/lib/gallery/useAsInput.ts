// Use as input (GA-10): from a file, a workspace workflow that takes its
// kind, opened with the argument filled in as `output:<name>`. The
// workflow page reads the fill once (WORKFLOW_PREFILL_KEY).
import { WORKFLOW_PREFILL_KEY } from '../chainActions'
import { mediaKindByName, type MediaKind } from '../mediaArguments'
import { OUTPUT, reference } from '../references'
import { go } from '../router.svelte'
import type { WorkflowCard } from '../types'
import { workspace } from '../workspace.svelte'

export type InputTarget = { workflow: string; variable: string }

/** The workspace's own workflows (not the read-only examples) with a
 * variable named for `kind`, the first such variable each, by name */
export function inputTargets(
  cards: Record<string, WorkflowCard>,
  kind: MediaKind,
): InputTarget[] {
  const targets: InputTarget[] = []
  for (const [workflow, card] of Object.entries(cards)) {
    if (card.writable === false) continue
    const variable = (card.variable_names ?? []).find(
      (name) => mediaKindByName(name) === kind,
    )
    if (variable !== undefined) targets.push({ workflow, variable })
  }
  return targets.sort((a, b) => a.workflow.localeCompare(b.workflow))
}

/** Open the workflow's page with `file` in its media variable */
export function useAsInput(target: InputTarget, file: string): void {
  sessionStorage.setItem(
    WORKFLOW_PREFILL_KEY,
    JSON.stringify({
      workflow: target.workflow,
      arguments: { [target.variable]: reference(OUTPUT, file) },
    }),
  )
  go('ws', workspace.current, 'workflows', ...target.workflow.split('/'))
}
