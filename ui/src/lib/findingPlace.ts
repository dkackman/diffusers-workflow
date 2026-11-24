// Where in the workflow a validation error sits, read off the path the
// server sent with it: `steps[<i>]…` is that step, `variables.<name>…` or
// `arguments.<name>…` that variable. Only a place the document on screen
// still has is one - an index past the end, or a name it does not declare,
// places nothing, so the editor never offers a Go to that lands nowhere.
import { VARIABLE_PATH } from './findings'

export type Place =
  | { kind: 'step'; index: number; name: string }
  | { kind: 'variable'; name: string }

const STEP_PATH = /^steps\[(\d+)\]/

export function findingPlace(
  path: string | null,
  doc: Record<string, any>,
): Place | null {
  if (!path) return null
  const step = STEP_PATH.exec(path)
  if (step) {
    const index = Number(step[1])
    const found = doc.steps?.[index]
    return found ? { kind: 'step', index, name: String(found.name) } : null
  }
  const variable = VARIABLE_PATH.exec(path)?.[1]
  return variable !== undefined && Object.hasOwn(doc.variables ?? {}, variable)
    ? { kind: 'variable', name: variable }
    : null
}
