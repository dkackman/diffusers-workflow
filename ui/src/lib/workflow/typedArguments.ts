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
      .map(([key, raw]) => [
        key,
        coerce(widgetFor(undefined, declared[key]), raw),
      ]),
  )
}
