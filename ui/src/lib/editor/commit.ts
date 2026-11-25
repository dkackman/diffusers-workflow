// What a form field commits into the document (#1656)
import {
  coerce,
  displayValue,
  finiteNumber,
  isReference,
  widgetFor,
} from '../editor'
import type { PipelineParameter } from '../types'

/** An empty commit on a null field leaves it null: Enter on an untouched
 * field is a commit, and '' is not null to the engine */
export function keepsNull(raw: string, previous: unknown): boolean {
  return previous === null && raw.trim() === ''
}

/** The value a step argument's field commits. A reference says nothing about
 * the type of the literal replacing it, so the parameter does: its default,
 * else its annotation. */
export function commitArgument(
  parameter: PipelineParameter | undefined,
  raw: string,
  previous: unknown,
): unknown {
  if (keepsNull(raw, previous)) return null
  const shape = isReference(previous)
    ? (parameter?.default ?? undefined)
    : previous
  return coerce(widgetFor(parameter, shape), raw)
}

// The keys whose string the engine loads as a type (dw/arguments.py);
// tests/test_ui_twins.py pins this against it
const TYPE_KEY = /(_type|_dtype)$|^dtype$/
const DOTTED_NAME = /^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$/

const isPlainWord = (value: unknown) =>
  typeof value === 'string' &&
  value !== '' &&
  !value.startsWith('{') &&
  !DOTTED_NAME.test(value)

/** What adding an argument writes: the parameter's default, null where that
 * is its default, and a plain word under a type key in {} so it stays a
 * string. A required argument, or one no schema names, starts empty. */
export function addedArgument(
  name: string,
  parameter: PipelineParameter | undefined,
): unknown {
  const value = parameter?.default
  if (value === null || value === undefined) {
    return parameter && !parameter.required ? null : ''
  }
  return TYPE_KEY.test(name) && isPlainWord(value) ? `{${value}}` : value
}

/** Write a number or a reference into target[key], or delete the key on
 * blank input. Other text is refused. Returns the text the input should now
 * show, so it never holds what the document does not. */
export function setNumber(
  target: Record<string, unknown>,
  key: string,
  raw: string,
): string {
  const text = raw.trim()
  const value = isReference(text) ? text : finiteNumber(text)
  if (!text) delete target[key]
  else if (value !== null) target[key] = value
  return displayValue(target[key])
}
