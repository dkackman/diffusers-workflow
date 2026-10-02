/** The engine's reference prefixes and the names that go with them.
 * dw/references.py owns them; tests/test_ui_twins.py pins this copy. The
 * one module in ui/src that spells a prefix: the UI ratchet's
 * prefix_literals counts any other. */

export const ASSET = 'asset:'
export const OUTPUT = 'output:'
export const PROMPT = 'prompt:'
export const VARIABLE = 'variable:'
export const PREVIOUS_RESULT = 'previous_result:'
export const CONSTANT = 'constant:'
export const ITEM = 'item:'
export const GATHER = 'gather:'
export const BUILTIN = 'builtin:'
export const CONSTRAINT = 'constraint:'

export const PREFIXES = [
  ASSET,
  OUTPUT,
  PROMPT,
  VARIABLE,
  PREVIOUS_RESULT,
  CONSTANT,
  ITEM,
  GATHER,
  BUILTIN,
  CONSTRAINT,
] as const

/** Joins a for_each step's name to an entry's: `<step>@<entry>`. */
export const MEMBER_SEPARATOR = '@'

/** The key a reference object names an earlier step under, unprefixed. */
export const FROM_PREVIOUS_RESULT_KEY = 'from_previous_result'

/** A value the engine resolves later, so it is always edited as text:
 * a widget that coerces it (a checkbox, a number) would replace it. */
export function isReference(value: unknown): value is string {
  return typeof value === 'string' && PREFIXES.some((p) => value.startsWith(p))
}

export type Prefix = (typeof PREFIXES)[number]

/** A reference to `name` under `prefix`: how every module writes one. */
export function reference(prefix: Prefix, name: string): string {
  return prefix + name
}

/** The name a reference names under `prefix`, trimmed as the engine trims
 * it, or null when the value is not that kind of reference. */
export function referenceName(value: unknown, prefix: Prefix): string | null {
  return typeof value === 'string' && value.startsWith(prefix)
    ? value.slice(prefix.length).trim()
    : null
}
