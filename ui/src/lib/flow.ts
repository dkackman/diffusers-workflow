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
        // A nameless consumer has nothing sensible to show as a chip -
        // still counted in resolvedRefs/inputs above, just not surfaced
        // as an edge on the producer
        if (step.name) graph[earlier.get(target)!].consumers.push(step.name)
      }
    })
  })
  return graph
}

/** Visit every string anywhere inside a JSON value, along with the dotted
 * path (array indices included) it was found at - e.g. `arguments.image`
 * or `loras.0.model_name`. */
function scanStringsWithPath(
  value: unknown,
  path: string[],
  visit: (s: string, path: string[]) => void,
): void {
  if (typeof value === 'string') visit(value, path)
  else if (Array.isArray(value))
    value.forEach((v, i) => scanStringsWithPath(v, [...path, String(i)], visit))
  else if (value !== null && typeof value === 'object')
    Object.entries(value).forEach(([k, v]) =>
      scanStringsWithPath(v, [...path, k], visit),
    )
}

/** The path's most specific, human-meaningful segment: the last one that
 * isn't a bare array index (so `loras.0.model_name` reads as
 * `model_name`, not `0`). */
function attributeLabel(path: string[]): string {
  for (let i = path.length - 1; i >= 0; i--) {
    if (!/^\d+$/.test(path[i])) return path[i]
  }
  return path[path.length - 1] ?? ''
}

export type StepKind = 'pipeline' | 'task' | 'workflow' | 'unknown'

export interface FlowNode {
  name: string
  index: number
  kind: StepKind
  /** A short label describing what the step runs - the pipeline's
   * component_type, the task's command, or the sub-workflow's path. */
  detail: string
  /** True when the step has no `previous_result` edge from an earlier
   * step - it starts a chain rather than continuing one. */
  isEntryPoint: boolean
  /** A statically-known count of items this step produces, when the
   * JSON says so directly (currently just a literal
   * `num_images_per_prompt`). Null means "unknown, assume 1". */
  producedCount: number | null
}

export interface FlowEdge {
  from: string
  to: string
  /** The argument/attribute name the reference was written under. */
  attribute: string
}

export interface DataFlowGraph {
  nodes: FlowNode[]
  edges: FlowEdge[]
  /** Steps with more than one distinct incoming previous_result producer -
   * a cartesian-product point invisible from reading the JSON step by
   * step. Keyed by step name. */
  fanIn: Map<string, { producers: string[]; label: string }>
}

function stepKindAndDetail(step: Record<string, any>): [StepKind, string] {
  if (step.pipeline) {
    const type = step.pipeline.configuration?.component_type
    return ['pipeline', type ?? 'pipeline']
  }
  if (step.task) {
    return ['task', step.task.command ?? 'task']
  }
  if (step.workflow) {
    return ['workflow', step.workflow.path ?? 'workflow']
  }
  return ['unknown', '']
}

/** A literal `num_images_per_prompt` on the step's own pipeline arguments,
 * when it is a plain number in the JSON (not a `variable:` reference,
 * which can't be resolved without running the workflow). */
function producedCount(step: Record<string, any>): number | null {
  const n = step.pipeline?.arguments?.num_images_per_prompt
  return typeof n === 'number' ? n : null
}

/** The read-only data-flow view's graph: one node per step, one edge per
 * `previous_result:<step>` reference (labeled with the attribute that
 * carries it), entry points flagged, and fan-in points - steps combining
 * more than one upstream producer, where CLAUDE.md's cartesian-product
 * gotcha (4 images x 3 masks = 12) applies - called out explicitly since
 * that multiplication is invisible reading the JSON step by step. */
export function dataFlowGraph(workflow: Record<string, any>): DataFlowGraph {
  const steps: Array<Record<string, any>> = workflow.steps ?? []
  const nodes: FlowNode[] = steps.map((step, index) => {
    const [kind, detail] = stepKindAndDetail(step)
    return {
      name: step.name ?? `step-${index}`,
      index,
      kind,
      detail,
      isEntryPoint: true,
      producedCount: producedCount(step),
    }
  })
  const edges: FlowEdge[] = []
  const incomingProducers = new Map<string, Set<string>>()

  steps.forEach((step, index) => {
    const earlier = new Map<string, number>()
    steps.slice(0, index).forEach((s, i) => {
      if (s.name) earlier.set(s.name, i)
    })
    const seen = new Set<string>() // producer|attribute - dedupe repeats
    scanStringsWithPath(step, [], (value, path) => {
      const target = refTarget(value)
      if (target === null || !earlier.has(target)) return
      const attribute = attributeLabel(path)
      const key = `${target}|${attribute}`
      if (seen.has(key)) return
      seen.add(key)
      edges.push({
        from: target,
        to: step.name ?? nodes[index].name,
        attribute,
      })
      nodes[index].isEntryPoint = false
      const producers = incomingProducers.get(step.name) ?? new Set()
      producers.add(target)
      incomingProducers.set(step.name, producers)
    })
  })

  const fanIn = new Map<string, { producers: string[]; label: string }>()
  for (const [stepName, producerSet] of incomingProducers) {
    if (producerSet.size < 2) continue
    const producers = [...producerSet]
    const counts = producers.map((p) => {
      const node = nodes.find((n) => n.name === p)
      return node?.producedCount ?? null
    })
    const label = counts.every((c) => c !== null)
      ? `${counts.join(' × ')} = ${counts.reduce((a, b) => (a ?? 1) * (b ?? 1), 1)}`
      : `combines ${producers.length} upstream steps`
    fanIn.set(stepName, { producers, label })
  }

  return { nodes, edges, fanIn }
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
