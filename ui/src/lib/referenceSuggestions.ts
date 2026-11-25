// The references a workflow's value inputs are offered, and the ones it
// makes that nothing declares.
import { danglingReferenceDetails, forEachReferences } from './flow'
import { PREVIOUS_RESULT, PROMPT, VARIABLE, reference } from './references'
import type { WorkflowDefinition, WorkflowStep } from './types'

/** `previous_result:` for a step, plus the `.frames` and `.audio` of a
 * video result. */
function resultReferences(step: WorkflowStep): string[] {
  const own = reference(PREVIOUS_RESULT, step.name)
  return (step.result?.content_type ?? '').startsWith('video')
    ? [own, `${own}.frames`, `${own}.audio`]
    : [own]
}

/** Reference completions available to a value input in step `stepIndex`:
 * every declared variable, every EARLIER step's result - with media
 * property suffixes for steps whose result is a video - every stored
 * prompt the library lists, and the for_each references the step can
 * make (`item:` and its fields, `gather:`, members). An earlier for_each
 * step is offered by plain name only where the engine allows it: from a
 * for_each over the same list. */
export function referenceSuggestions(
  workflow: WorkflowDefinition,
  stepIndex: number,
  promptNames: string[] = [],
): string[] {
  const suggestions: string[] = []
  for (const name of Object.keys(workflow.variables ?? {})) {
    suggestions.push(reference(VARIABLE, name))
  }
  for (const name of promptNames) {
    suggestions.push(reference(PROMPT, name))
  }
  const forEach = forEachReferences(workflow, stepIndex)
  const steps = workflow.steps ?? []
  for (const step of steps.slice(0, Math.max(0, stepIndex))) {
    if (step.name && !forEach.plainRefused.has(step.name))
      suggestions.push(...resultReferences(step))
  }
  suggestions.push(...forEach.suggestions)
  return suggestions
}

/** References the workflow makes that nothing declares: variable: names
 * missing from variables, previous_result: names that match no EARLIER
 * step, and - when the prompt library's listing is supplied - prompt:
 * names it does not hold. The engine would fail these at run time; the
 * editor says so now. */
export function danglingReferences(
  workflow: WorkflowDefinition,
  promptNames?: string[],
): string[] {
  return danglingReferenceDetails(workflow, promptNames).map((d) => d.message)
}
