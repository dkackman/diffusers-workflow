// Renaming a workflow step, and every reference to it with it. A step is
// named by `previous_result:<step>`, `gather:<step>` and a bare
// `from_previous_result` value, each either exactly the name or the name
// followed by a `.property` or an `@entry` member (docs/WORKFLOW_GUIDE.md,
// *References*). A value that only starts with the name - `<step>x` - names
// another step and is left alone.
import {
  FROM_PREVIOUS_RESULT_KEY,
  GATHER,
  MEMBER_SEPARATOR,
  PREVIOUS_RESULT,
  isReference,
} from './references'

const PROPERTY_SEPARATOR = '.'

/** Why `name` cannot name a step, in the editor's words, or null. A space,
 * `.` or `@` would be read as part of a reference (`<step>.<property>`,
 * `<step>@<entry>`); `others` are the names the other steps hold. */
export function stepNameProblem(name: string, others: string[]): string | null {
  if (name === '' || /[\s.@]/.test(name))
    return "A step name can't be empty or hold a space, . or @."
  return others.includes(name)
    ? `Another step is already called ${name}.`
    : null
}

/** `rest` with a leading `from` swapped for `to`, when `from` is all of it
 * or is followed by a property or member separator; null otherwise. */
function renamed(rest: string, from: string, to: string): string | null {
  if (rest === from) return to
  const next = rest.charAt(from.length)
  return rest.startsWith(from) &&
    (next === PROPERTY_SEPARATOR || next === MEMBER_SEPARATOR)
    ? to + rest.slice(from.length)
    : null
}

/** A reference string with the step renamed, or null when it names another
 * step or is no step reference. `bare` is a `from_previous_result` value. */
function renamedReference(
  value: string,
  bare: boolean,
  from: string,
  to: string,
): string | null {
  if (bare && !isReference(value)) return renamed(value, from, to)
  for (const prefix of [PREVIOUS_RESULT, GATHER]) {
    if (!value.startsWith(prefix)) continue
    const rest = renamed(value.slice(prefix.length), from, to)
    return rest === null ? null : prefix + rest
  }
  return null
}

/** A copy of `workflow` with step `from` called `to` and every reference
 * to it - in the variables and in every step, at any depth - following it.
 * `count` is the references changed, the step's own name not among them. */
export function renameStep(
  workflow: Record<string, any>,
  from: string,
  to: string,
): { workflow: Record<string, any>; count: number } {
  let count = 0
  const walk = (value: unknown, key: string | null): unknown => {
    if (Array.isArray(value)) return value.map((v) => walk(v, null))
    if (value !== null && typeof value === 'object')
      return Object.fromEntries(
        Object.entries(value).map(([k, v]) => [k, walk(v, k)]),
      )
    if (typeof value !== 'string') return value
    const next = renamedReference(
      value,
      key === FROM_PREVIOUS_RESULT_KEY,
      from,
      to,
    )
    if (next === null) return value
    count += 1
    return next
  }
  const copy = walk(workflow, null) as Record<string, any>
  const step = (copy.steps ?? []).find(
    (s: Record<string, any>) => s?.name === from,
  )
  if (step) step.name = to
  return { workflow: copy, count }
}
