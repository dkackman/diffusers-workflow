// What a form sends: each typed value as the type its variable declares, so a
// number default sends a number and a list default sends a list. A reference
// (`output:`, `asset:`, `variable:`) stays the string it is.
import { coerce, widgetFor } from '../editor'

/** The overrides with every blank dropped (a blank leaves the workflow's own
 * default in place), each of the rest coerced to the type of its declared
 * default. A key with no declared default stays text. */
export function typedArguments(
  overrides: Record<string, string>,
  declared: Record<string, unknown>,
): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(overrides)
      .filter(([, raw]) => raw.trim() !== '')
      .map(([key, raw]) => [key, typed(raw, declared[key])]),
  )
}

function typed(raw: string, fallback: unknown): unknown {
  if (fallback === null) return container(raw) ?? raw
  const value = coerce(widgetFor(undefined, fallback), raw)
  // A list default takes a list. A bare `5` parses as a number, which the
  // engine cannot make a list of; as text it is split into one
  if (Array.isArray(fallback) && !Array.isArray(value)) return raw
  return value
}

/** A typed JSON list or object, for a variable declared null (a script's
 * `lines`, a cut's `shots`): the declaration says nothing of the type, so
 * only what is plainly a container is read as one. Undefined otherwise */
function container(raw: string): unknown {
  const text = raw.trim()
  if (!text.startsWith('[') && !text.startsWith('{')) return undefined
  try {
    return JSON.parse(text) as unknown
  } catch {
    return undefined
  }
}
