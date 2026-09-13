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

/** The for_each members a run has finished, in the engine's own
 * `group@entry` spelling - the names the flow view's member chips carry.
 * Only top-level ends count: a sub-workflow's inner members, whose
 * `step_end` carries the composed step as `parent_step`, belong to no
 * chip this graph draws. */
export function finishedMembers(events: JobEvent[]): string[] {
  return events
    .filter((event) => event.event === 'step_end' && !event.parent_step)
    .map((event) => event.step as string)
    .filter((step) => step.includes(MEMBER_SEPARATOR))
}

/** The for_each member the run is on right now, or undefined when the
 * last step it started is not a member - or is a member that has already
 * ended, since between one entry finishing and the next starting the run
 * is on neither, and a chip still amber there would lie about progress. A
 * sub-workflow's inner steps do not count, member-named or not, for the
 * same reason as above. */
export function activeMember(events: JobEvent[]): string | undefined {
  const ended = new Set(finishedMembers(events))
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i]
    if (event.event !== 'step_start' || event.parent_step) continue
    const step = event.step as string
    if (!step.includes(MEMBER_SEPARATOR) || ended.has(step)) return undefined
    return step
  }
  return undefined
}
