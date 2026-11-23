/** What the chain segment table offers on a row and above it (#1476).
 *
 * **Repair this segment** opens `repair-segment` on the row's segment. Only
 * one family has that template, and the UI never names the family: the
 * chain step's `pipeline.from_pretrained_arguments.model_name` (a literal or
 * a `variable:` the run resolved) has to be a model a pipeline step of the
 * repair template loads. Rows 2..N of a take whose segments were kept can be
 * repaired; every other row says, in one sentence, why not.
 *
 * **Continue the chain** reruns a take that stopped short of its plan from
 * its last kept segment: the workflow has to declare the variables its
 * chain's `stop_after` and `resume_from` name, and every row has to have its
 * segment file. Pure but for `repairTemplateModels`, the one read, so the
 * rules are tested without a DOM. */

import type { ChainRow, ChainTable } from './chainRows'
import { definitionStep, resolved } from './chainRows'
import { OUTPUT, VARIABLE, reference, referenceName } from './references'
import { api, type ValidateRequest } from './api'
import type { WorkflowDefinition } from './types'

/** The only repair template the catalog holds */
export const REPAIR_TEMPLATE = 'templates/minimax/repair-segment'

/** The sessionStorage key the workflow page reads its pre-filled values from,
 * once: `{workflow, arguments}` */
export const WORKFLOW_PREFILL_KEY = 'dw-workflow-prefill'

export const NO_REPAIR_TEMPLATE =
  'no repair-segment template for this model family'
export const FIRST_SEGMENT =
  'segment 1 cannot be repaired - re-render with a new seed'
export const REPAIRED_TAKE = 'a repaired take has no segment files'
export const NO_SEGMENTS = 'this run did not keep its segments'

type Definition = Record<string, any> | null | undefined

export type RepairAction =
  | { enabled: true; workflow: string; arguments: Record<string, string> }
  | { enabled: false; reason: string }

export interface ContinueAction {
  arguments: Record<string, string | null>
}

const modelOf = (
  step: any,
  definition: Definition,
  args?: Record<string, unknown> | null,
): string | null => {
  const pipeline = step?.pipeline
  const name = resolved(
    pipeline?.from_pretrained_arguments?.model_name,
    definition,
    args,
  )
  return typeof name === 'string' && name ? name : null
}

/** The model the chained step loads, as the run resolved it; null for a
 * step the definition does not hold at its top level, or names no model of */
export function chainModel(
  definition: Definition,
  step: string,
  args?: Record<string, unknown> | null,
): string | null {
  return modelOf(definitionStep(definition, step), definition, args)
}

/** Every model a pipeline step of the repair template loads */
export function repairModels(repair: Definition): string[] {
  const steps = Array.isArray(repair?.steps) ? repair.steps : []
  return steps
    .map((step: any) => modelOf(step, repair, null))
    .filter((name: string | null): name is string => name !== null)
}

/** The models the repair template loads, as the server has it now; none
 * when this server has no repair template */
export async function repairTemplateModels(): Promise<string[]> {
  try {
    const { definition } = await api.getWorkflow(REPAIR_TEMPLATE)
    return repairModels(definition as Definition)
  } catch {
    return []
  }
}

/** Whether, and how, one row can be repaired. `model` is the chain step's
 * (`chainModel`), `models` the repair template's (`repairModels`; empty
 * when it could not be read) */
export function repairAction(
  table: ChainTable,
  row: ChainRow,
  model: string | null,
  models: readonly string[],
): RepairAction {
  if (model === null || !models.includes(model))
    return { enabled: false, reason: NO_REPAIR_TEMPLATE }
  if (row.index === 1) return { enabled: false, reason: FIRST_SEGMENT }
  if (table.rows.some((r) => r.repaired))
    return { enabled: false, reason: REPAIRED_TAKE }
  if (!row.segmentFile || !table.take)
    return { enabled: false, reason: NO_SEGMENTS }
  return {
    enabled: true,
    workflow: REPAIR_TEMPLATE,
    arguments: {
      chain_take: reference(OUTPUT, table.take),
      segment: String(row.index),
    },
  }
}

/** The variable a chain setting names, when the workflow declares it */
function declaredVariable(value: unknown, definition: Definition) {
  const name = referenceName(value, VARIABLE)
  const variables = definition?.variables
  return name !== null &&
    variables &&
    typeof variables === 'object' &&
    name in variables
    ? name
    : null
}

/** Whether a take stopped before its planned length with every segment it
 * made on file */
function stoppedShort({ rows, planned }: ChainTable): boolean {
  if (!rows.length || planned === null || rows.length >= planned) return false
  return rows.every((row) => row.segmentFile)
}

/** The rerun arguments that continue a stopped take from its last segment,
 * or null when it is not offered */
export function continueAction(
  table: ChainTable,
  definition: Definition,
): ContinueAction | null {
  const { rows } = table
  if (!stoppedShort(table)) return null
  const chain = definitionStep(definition, table.step)?.pipeline?.chain
  if (!chain || typeof chain !== 'object') return null
  const stopAfter = declaredVariable(chain.stop_after, definition)
  const resumeFrom = declaredVariable(chain.resume_from, definition)
  if (!stopAfter || !resumeFrom) return null
  const last = rows[rows.length - 1].segmentFile as string
  return {
    arguments: {
      [resumeFrom]: reference(OUTPUT, last),
      [stopAfter]: null,
    },
  }
}

/** What `/api/validate` is asked to price a Continue: the rerun's merged
 * arguments (the job's own, then the action's), in the job's workspace, by
 * the catalog name it ran as when it has one (so the plan quotes that
 * workflow's observed figures), otherwise the definition it ran inline. Null
 * when there is neither, and the confirm says it has no estimate. */
export function continueValidation(
  job: {
    workflow_name?: string | null
    workspace: string
    arguments?: Record<string, unknown> | null
  },
  action: ContinueAction,
  definition: Definition,
): ValidateRequest | null {
  const source = job.workflow_name
    ? { workflow_path: job.workflow_name }
    : definition
      ? { workflow: definition as WorkflowDefinition }
      : null
  if (!source) return null
  return {
    ...source,
    workspace: job.workspace,
    arguments: { ...(job.arguments ?? {}), ...action.arguments },
    sizes: false,
  }
}

/** The Continue confirm's words: what it reruns, then the estimate line. */
export function continueConfirmText(
  table: ChainTable,
  estimate: string,
): string {
  const kept = table.rows.length
  const remaining = (table.planned ?? 0) - kept
  return (
    `Queue this job again from its last kept segment (${kept} of ${table.planned})? ` +
    `It renders the remaining ${remaining} segment${remaining === 1 ? '' : 's'} of ${table.step}.` +
    `\n\nEstimate: ${estimate}`
  )
}
