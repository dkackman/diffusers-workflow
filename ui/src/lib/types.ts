import type { components } from './generated/api-schema'

type Schemas = components['schemas']

// Server responses: generated from the server's response models
// (dw/server/api_models.py), so a field the server stops sending fails the
// type check rather than reading undefined at runtime
export type HealthInfo = Schemas['HealthInfo']
export type ServerAddress = Schemas['ServerAddress']
export type ServerInfo = Schemas['ServerInfo']
export type MemoryInfo = Schemas['MemoryInfo']
export type ModelRevision = Schemas['ModelRevision']
export type ModelRepo = Schemas['ModelRepo']
export type ModelCache = Schemas['ModelCache']
export type ModelDownload = Schemas['ModelDownload']
export type DiffusersStatus = Schemas['DiffusersStatus']
export type JobSummary = Schemas['JobSummary']
export type ManifestEntry = Schemas['ManifestEntry']
export type JobDetail = Schemas['JobDetail']
export type AcknowledgedCost = Schemas['AcknowledgedCost']
export type JobProgress = Schemas['JobProgress']
export type JobList = Schemas['JobList']
export type JobWorkflow = Schemas['JobWorkflow']
export type JobExport = Schemas['JobExport']
export type JobMoved = Schemas['JobMoved']
export type JobCancelled = Schemas['JobCancelled']
export type RunDeleted = Schemas['RunDeleted']
export type ValidationResult = Schemas['ValidationResult']
export type Plan = Schemas['Plan']
export type PlanEstimate = Schemas['PlanEstimate']
export type RequiredDownload = Schemas['RequiredDownload']
export type ElidedStep = Schemas['ElidedStep']
export type WorkflowCost = Schemas['WorkflowCost']
export type LibraryRoot = Schemas['LibraryRoot']
export type ShadowedEntry = Schemas['ShadowedEntry']
export type PromptDetail = Schemas['PromptCard']
export type EnhancerPreset = Schemas['EnhancerPreset']
export type WorkflowCard = Schemas['WorkflowCard']
export type WorkflowList = Schemas['WorkflowList']
export type WorkflowSaved = Schemas['WorkflowSaved']
export type PromptList = Schemas['PromptList']
export type WorkspaceInfo = Schemas['WorkspaceInfo']
export type WorkspaceList = Schemas['WorkspaceList']
export type WorkspaceDeleted = Schemas['WorkspaceDeleted']
export type Deleted = Schemas['Deleted']
export type WorkflowDeleted = Schemas['WorkflowDeleted']
export type PromptSaved = Schemas['PromptSaved']
export type EnhancerPresets = Schemas['EnhancerPresets']

export interface JobEvent {
  seq: number
  event: string
  [key: string]: unknown
}

/** The `step_end` event, as the job page reads it: what the step saved and
 * where, before the manifest confirms it. */
export interface StepEndEvent extends JobEvent {
  event: 'step_end'
  step: string
  files?: string[]
  subfolder?: string
  reused?: boolean
}

/** What a workflow makes. The server derives it from the definition
 * (dw/server/catalog_shape.py) unless the file declares one. */
export const WORKFLOW_SHAPES = [
  'image',
  'image-set',
  'image-edit',
  'shot',
  'sequence',
  'audio',
  'text',
  'utility',
] as const
export type WorkflowShape = (typeof WORKFLOW_SHAPES)[number]

/** Independent facts about how the output is made or what it needs. */
export const WORKFLOW_TRAITS = [
  'has-audio',
  'chained',
  'image-conditioned',
  'identity-referenced',
  'needs-input-media',
  'composes-workflows',
] as const
export type WorkflowTrait = (typeof WORKFLOW_TRAITS)[number]

export interface WorkflowDefinition {
  id: string
  variables?: Record<string, unknown>
  steps?: Array<Record<string, unknown>>
  [key: string]: unknown
}

/** A workflow plus where it came from - `getWorkflow` reads these off the
 * `X-Workflow-Origin` / `X-Workflow-Writable` response headers. Beside the
 * definition rather than spread into it, as for a prompt: a workflow is
 * validated and saved back exactly as it was read, and a stray root field
 * fails the schema - the engine refuses unknown root keys rather than
 * ignoring them. */
export interface WorkflowWithOrigin {
  definition: WorkflowDefinition
  /** 'workspace' | 'examples' | 'builtin'. */
  origin: string
  writable: boolean
}

export interface PromptDefinition {
  text: string
  description?: string
  intended_model?: string
  negative_prompt?: string
  tags?: string[]
  enhanced?: { model?: string; idea?: string }
}

/** A stored prompt plus which library it came from - `getPrompt` reads the
 * two off the `X-Prompt-Origin` / `X-Prompt-Writable` response headers, the
 * way `getWorkflow` reads a workflow's source. Beside the definition rather
 * than spread into it: a prompt is saved back exactly as it was read, and a
 * stray field would fail the schema. */
export interface StoredPrompt {
  prompt: PromptDefinition
  /** 'workspace' | 'examples'. */
  origin: string
  writable: boolean
}

export interface PipelineParameter {
  name: string
  required: boolean
  default: unknown
  annotation: string | null
  doc_type?: string
  description?: string
}

export interface PipelineDescription {
  name: string
  summary: string
  accepts_kwargs: boolean
  parameters: PipelineParameter[]
  compatibles?: string[]
}

export interface GalleryFile {
  name: string
  folder: string
  /** What followed the run id in the file's path - the `final` /
   * `intermediate` a step's `result.subfolder` chose, `''` for none. */
  subfolder: string
  /** The run that wrote the file, `''` under the flat layout. */
  run_id: string
  /** That run's ordinal among the workflow's runs - what the grid shows as
   * `v4`. Two runs write the same `label`, so this is what tells them
   * apart at a glance. Assigned when the run opens and never renumbered,
   * so a deleted sibling leaves a gap. Null when there is no run. */
  version: number | null
  url: string
  kind: 'image' | 'video' | 'audio'
  size: number
  mtime: number
  label: string
}

/** One file in the asset library - the input media an `asset:` reference
 * names. Reported by reference rather than by path, so a client never has
 * to build one. */
export interface AssetFile {
  name: string
  reference: string
  folder: string
  kind: 'image' | 'video' | 'audio'
  size: number
  mtime: number
  /** Which library it came from: this workspace's own, the `common` one
   * every workspace shares, or a read-only examples tree. The last is why
   * a delete can answer 403. */
  origin: 'workspace' | 'common' | 'examples'
  url: string
}

export interface AssetLibrary extends LibraryRoot {
  origin: AssetFile['origin']
}

/** An asset a nearer library hides: same shape as `AssetFile` except there
 * is no `url` - that URL would serve the shadowing file, not this one - and
 * `shadowed_by` names the origin that won. */
export type ShadowedAsset = Omit<AssetFile, 'url'> & {
  shadowed_by: AssetFile['origin']
}
