import { booleanOf } from '../editor'
import {
  VARIABLE_NAME_CHARACTERS,
  VARIABLE_NAME_FIRST,
  isReference,
} from '../references'

export type VariableType = 'text' | 'number' | 'boolean' | 'json' | 'none'

export const VARIABLE_TYPES: VariableType[] = [
  'text',
  'number',
  'boolean',
  'json',
  'none',
]

/** Why a new variable name is refused, in the editor's words, or null.
 * The engine's rule (`validate_variable_name`): a letter or `_` first, then
 * `VARIABLE_NAME_CHARACTERS`. */
export function variableNameProblem(
  name: string,
  taken: string[],
): string | null {
  const allowed = [...name].every((c) => VARIABLE_NAME_CHARACTERS.includes(c))
  if (!name || !VARIABLE_NAME_FIRST.includes(name[0]) || !allowed) {
    return 'A variable name starts with a letter or _ and holds letters, numbers, _ and -.'
  }
  return taken.includes(name)
    ? `There is already a variable called ${name}.`
    : null
}

export function typeOf(value: unknown): VariableType {
  if (value === null) return 'none'
  if (typeof value === 'object') return 'json'
  if (typeof value === 'boolean') return 'boolean'
  if (typeof value === 'number') return 'number'
  return 'text'
}

function parsed(raw: string): unknown {
  try {
    return JSON.parse(raw)
  } catch {
    return raw
  }
}

function finiteNumber(raw: string): number | null {
  const n = Number(raw)
  return raw.trim() !== '' && Number.isFinite(n) ? n : null
}

/** What a new variable's first typed value means: true/false, a finite
 * number, null, or JSON that opens with { or [ and parses; anything else,
 * a reference included, is the text as typed. */
export function inferValue(raw: string): unknown {
  if (isReference(raw)) return raw
  if (raw === 'true') return true
  if (raw === 'false') return false
  if (raw === 'null') return null
  const number = finiteNumber(raw)
  if (number !== null) return number
  return /^\s*[{[]/.test(raw) ? parsed(raw) : raw
}

/** The typed text as a value of `type`; text that does not fit the type
 * (a reference, a half-typed number, a boolean that is not true or false)
 * stays text rather than being lost, and the row's type shows it. */
export function asType(raw: string, type: VariableType): unknown {
  if (type === 'none') return null
  if (type === 'boolean') return booleanOf(raw)
  if (type === 'number')
    return isReference(raw) ? raw : (finiteNumber(raw) ?? raw)
  if (type === 'json') return parsed(raw)
  return raw
}
