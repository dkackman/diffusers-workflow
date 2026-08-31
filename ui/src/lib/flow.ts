/** The workflow's data-flow graph. Execution is linear (steps run in
 * order) but data flow is a DAG: several steps can feed one consumer
 * (3 images -> 1 video) and one step can feed several. Everything the
 * editor says about flow - chips, highlights, reorder warnings, the
 * cartesian-product note - derives from here, and only from here. */

export interface StepFlow {
  name: string
  inputs: string[]
  consumers: string[]
  resolvedRefs: number
}

export interface DanglingReference {
  stepIndex: number
  message: string
}

/** Visit every string anywhere inside a JSON value. */
function scanStrings(value: unknown, visit: (s: string) => void): void {
  if (typeof value === 'string') visit(value)
  else if (Array.isArray(value)) value.forEach((v) => scanStrings(v, visit))
  else if (value !== null && typeof value === 'object')
    Object.values(value).forEach((v) => scanStrings(v, visit))
}

/** previous_result: target step name, media suffix stripped. */
function refTarget(s: string): string | null {
  if (!s.startsWith('previous_result:')) return null
  return s.slice('previous_result:'.length).split('.')[0]
}

export function flowGraph(workflow: Record<string, any>): StepFlow[] {
  const steps: Array<Record<string, any>> = workflow.steps ?? []
  const graph: StepFlow[] = steps.map((s) => ({
    name: s.name ?? '',
    inputs: [],
    consumers: [],
    resolvedRefs: 0,
  }))

  steps.forEach((step, index) => {
    // Only EARLIER steps are producers - the engine resolves in order
    const earlier = new Map<string, number>()
    steps.slice(0, index).forEach((s, i) => {
      if (s.name) earlier.set(s.name, i)
    })
    scanStrings(step, (s) => {
      const target = refTarget(s)
      if (target === null || !earlier.has(target)) return
      graph[index].resolvedRefs += 1
      if (!graph[index].inputs.includes(target)) {
        graph[index].inputs.push(target)
        graph[earlier.get(target)!].consumers.push(step.name ?? '')
      }
    })
  })
  return graph
}

export function danglingReferenceDetails(
  workflow: Record<string, any>,
  promptNames?: string[],
): DanglingReference[] {
  const problems: DanglingReference[] = []
  const variables = new Set(Object.keys(workflow.variables ?? {}))
  const prompts = promptNames === undefined ? null : new Set(promptNames)
  const steps: Array<Record<string, any>> = workflow.steps ?? []

  steps.forEach((step, index) => {
    const earlier = new Set(
      steps
        .slice(0, index)
        .map((s) => s.name)
        .filter(Boolean),
    )
    scanStrings(step, (value) => {
      if (value.startsWith('variable:')) {
        const name = value.slice('variable:'.length)
        if (!variables.has(name)) {
          problems.push({
            stepIndex: index,
            message: `Step '${step.name}': variable:${name} - no such variable is declared`,
          })
        }
      } else if (value.startsWith('previous_result:')) {
        const name = refTarget(value)!
        if (!earlier.has(name)) {
          problems.push({
            stepIndex: index,
            message: `Step '${step.name}': previous_result:${name} - no earlier step has that name`,
          })
        }
      } else if (value.startsWith('prompt:')) {
        // Without a listing the server resolves these at run time - only
        // a supplied library can say a name is missing
        const name = value.slice('prompt:'.length)
        if (prompts !== null && !prompts.has(name)) {
          problems.push({
            stepIndex: index,
            message: `Step '${step.name}': prompt:${name} - the prompt library has no such prompt`,
          })
        }
      }
    })
  })
  return problems
}
