/** What made a file, as rows: the recipe RecipePanel shows. An image's
 * come from what it embeds (outputMeta.ts reads it); a clip embeds
 * nothing, so its come from the job that made it. The order and labels are
 * the plan's: workflow, job, step, model, seed, prompts, size, length, then
 * every other argument, each once. */

import { valueText } from './compare'
import { formatDuration } from './format'
import { DESCRIBED_ARGUMENTS, describeOutputMeta } from './outputMeta'
import { jobHref } from './routes'
import { workflowName } from './terms'
import type { GalleryMetadata, JobDetail } from './types'

export type RecipeRow = {
  key:
    | 'workflow'
    | 'job'
    | 'step'
    | 'model'
    | 'seed'
    | 'prompt'
    | 'negative'
    | 'size'
    | 'length'
    | 'argument'
  label: string
  value: string
  /** A value worth taking elsewhere: a copy button beside it */
  copy?: boolean
  href?: string
}

type Fields = Record<string, unknown>

const text = (v: unknown) => (typeof v === 'string' && v ? v : null)
const num = (v: unknown) => (typeof v === 'number' ? v : null)

/** The workflow's name: the job's file name when one is on record, else
 * the id inside the embedded definition - the only name an image carries */
function workflowOf(meta: GalleryMetadata, job: JobDetail | null) {
  const ref = meta.job as Fields | null
  const named = job ?? {
    workflow_name: text(ref?.workflow_name),
    workflow: text(ref?.workflow),
  }
  if (named.workflow_name || named.workflow) return workflowName(named)
  const embedded = meta.metadata?.workflow as Fields | undefined
  return text(embedded?.id) ?? ''
}

/** The rows a file's media gives: its pixel size, its length */
function mediaRows(media: Fields | null): RecipeRow[] {
  const rows: RecipeRow[] = []
  const w = num(media?.width)
  const h = num(media?.height)
  if (w !== null && h !== null)
    rows.push({ key: 'size', label: 'Size', value: `${w} × ${h}` })
  const length = num(media?.duration_seconds)
  if (length !== null)
    rows.push({ key: 'length', label: 'Length', value: formatDuration(length) })
  return rows
}

/** What the recipe's own rows read: the embedded recipe, else the job's
 * arguments */
function describe(meta: GalleryMetadata, job: JobDetail | null) {
  const embedded = meta.metadata
  return describeOutputMeta(
    embedded ?? (job ? { arguments: job.arguments } : null),
  )
}

/** What ran: the workflow, the job, the step and the model */
function ranRows(
  meta: GalleryMetadata,
  job: JobDetail | null,
  workspace: string,
): RecipeRow[] {
  const jobId = meta.job?.id ?? job?.id ?? ''
  return [
    { key: 'workflow', label: 'Workflow', value: workflowOf(meta, job) },
    { key: 'job', label: 'Job', value: jobId, href: jobHref(jobId, workspace) },
    {
      key: 'step',
      label: 'Step',
      value: text(meta.metadata?.step_name) ?? '',
    },
    { key: 'model', label: 'Model', value: describe(meta, job).model },
  ]
}

/** The values worth taking elsewhere: the seed and the prompts */
function copyRows(meta: GalleryMetadata, job: JobDetail | null): RecipeRow[] {
  const described = describe(meta, job)
  const seed = described.seed
  return [
    {
      key: 'seed',
      label: 'Seed',
      value: seed === undefined || seed === null ? '' : valueText(seed),
    },
    { key: 'prompt', label: 'Prompt', value: described.prompt },
    {
      key: 'negative',
      label: 'Negative prompt',
      value: described.negativePrompt,
    },
  ].map((row): RecipeRow => ({ ...(row as RecipeRow), copy: true }))
}

/** Every argument not already a row, in the order the file lists them */
function argumentRows(meta: GalleryMetadata, job: JobDetail | null) {
  const args = ((meta.metadata?.arguments as Fields | undefined) ??
    job?.arguments ??
    {}) as Fields
  return Object.entries(args)
    .filter(([key]) => !DESCRIBED_ARGUMENTS.includes(key))
    .map(([key, value]): RecipeRow => ({
      key: 'argument',
      label: key,
      value: valueText(value),
    }))
}

/** What made the file, in the plan's order. `job` is the job read when the
 * file embeds nothing; its link names `workspace`, the file's own (#1600) */
export function recipeRows(
  meta: GalleryMetadata,
  job: JobDetail | null,
  workspace: string,
): RecipeRow[] {
  const head = [...ranRows(meta, job, workspace), ...copyRows(meta, job)]
  return [
    ...head.filter((row) => row.value !== ''),
    ...mediaRows(meta.media),
    ...argumentRows(meta, job),
  ]
}

/** `Saved from <workflow file name> v<n>` for an asset saved from a job,
 * else '' */
export function savedFrom(meta: GalleryMetadata): string {
  const ref = meta.job as Fields | null
  if (!ref) return ''
  const named = text(ref.workflow_name) || text(ref.workflow)
  const workflow = named
    ? workflowName({ workflow_name: named })
    : String(ref.id)
  return `Saved from ${workflow}${meta.version ? ` v${meta.version}` : ''}`
}
