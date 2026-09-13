/** Where a run's event stream meets the definition it is running.
 *
 * The flow graph is drawn from the definition, where `for_each` is left
 * unexpanded and a sub-workflow is one step, but the engine *runs* the
 * definition expanded: it reports `base@open` where the definition says
 * `base`, and a sub-workflow's inner step under its own name where the
 * definition says the composed step. Nothing the page highlights matches
 * until both sides are reduced to the same name - which is what broke the
 * flow view's run state when list-driven steps arrived. */

import type { JobEvent } from './types'

/** The for_each step/member separator - `dw/for_each.py`'s
 * MEMBER_SEPARATOR, reserved in every step name, so the first one in a
 * name always splits a member from its group. */
const MEMBER_SEPARATOR = '@'

/** The name the definition gives the step the engine ran.
 *
 * A for_each member reduces to its group (`base@open` -> `base`); anything
 * a sub-workflow emitted reduces to the step that queued it, which the
 * engine puts on every event a child emits as `parent_step`
 * (`_parent_progress_fields`, dw/workflow.py). A plain step is its own
 * name, returned as it is. */
export function flowNodeName(
  step?: string,
  parentStep?: string,
): string | undefined {
  const name = parentStep || step
  if (!name) return undefined
  const at = name.indexOf(MEMBER_SEPARATOR)
  return at === -1 ? name : name.slice(0, at)
}

/** The nodes a run has finished, given the step names its `workflow_start`
 * listed.
 *
 * A for_each group is one node, so it is done only once every member is -
 * greening it on the first would say the group is behind us while its
 * remaining entries are still queued. A sub-workflow is done when the
 * composed step itself ends, not when one of the child's inner steps does,
 * so only the top-level step_end events count: a child's carry the parent's
 * name in `parent_step`, which is exactly the marker to leave out. */
export function finishedNodes(
  events: JobEvent[],
  stepNames: string[],
): string[] {
  const ended = new Set(
    events
      .filter((event) => event.event === 'step_end' && !event.parent_step)
      .map((event) => event.step as string),
  )
  const members = new Map<string, string[]>()
  for (const name of stepNames) {
    const node = flowNodeName(name)
    if (node) members.set(node, [...(members.get(node) ?? []), name])
  }
  return [...members]
    .filter(([, names]) => names.every((name) => ended.has(name)))
    .map(([node]) => node)
}
