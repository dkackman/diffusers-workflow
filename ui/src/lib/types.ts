import type { components } from './generated/api-schema'

type Schemas = components['schemas']

// Server responses: generated from the server's response models
// (dw/server/api_models.py), so a field the server stops sending fails the
// type check rather than reading undefined at runtime
export type StepTiming = Schemas['StepTiming']
export type HealthInfo = Schemas['HealthInfo']
export type WorkerInfo = Schemas['WorkerInfo']
export type ServerAddress = Schemas['ServerAddress']
export type ServerInfo = Schemas['ServerInfo']
export type MemoryInfo = Schemas['MemoryStatus']
export type ModelRevision = Schemas['ModelRevision']
export type ModelRepo = Schemas['ModelRepo']
export type ModelCache = Schemas['ModelCache']
export type ModelDownload = Schemas['ModelDownload']
export type ModelDownloads = Schemas['ModelDownloads']
export type ModelDeleted = Schemas['ModelDeleted']
export type DiffusersStatus = Schemas['DiffusersStatus']
export type DwUpdateState = Schemas['DwUpdateState']
export type DiskCacheInfo = Schemas['DiskCacheInfo']
export type DiskCacheList = Schemas['DiskCacheList']
export type DiskCacheCleared = Schemas['DiskCacheCleared']
export type DiskCacheEntry = Schemas['DiskCacheEntry']
export type DiskCacheEntryDeleted = Schemas['DiskCacheEntryDeleted']
export type JobSummary = Schemas['JobSummary']
export type ManifestEntry = Schemas['ManifestEntry']
export type JobDetail = Schemas['JobDetail']
export type AcknowledgedCost = Schemas['AcknowledgedCost']
export type JobProgress = Schemas['JobProgress']
export type JobList = Schemas['JobList']
export type JobWorkflow = Schemas['JobWorkflow']
export type RestartBatchRerun = Schemas['RestartBatchRerun']
export type RestartBatchSize = Schemas['RestartBatchSize']
export type JobExport = Schemas['JobExport']
export type JobMoved = Schemas['JobMoved']
export type JobCancelled = Schemas['JobCancelled']
export type RunDeleted = Schemas['RunDeleted']
export type ValidationResult = Schemas['ValidationResult']
export type Plan = Schemas['Plan']
export type PlanEstimate = Schemas['PlanEstimate']
export type PlanHostMemory = Schemas['PlanHostMemory']
export type RequiredDownload = Schemas['RequiredDownload']
export type ElidedStep = Schemas['ElidedStep']
export type WorkflowCost = Schemas['WorkflowCost']
export type LibraryRoot = Schemas['LibraryRoot']
export type ShadowedEntry = Schemas['ShadowedEntry']
export type PromptDetail = Schemas['PromptCard']
export type EnhancerPreset = Schemas['EnhancerPreset']
export type WorkflowCard = Schemas['WorkflowCard']
/** What a variable that feeds a media argument takes. */
export type InputKind = NonNullable<WorkflowCard['inputs']>[string]
export type WorkflowList = Schemas['WorkflowList']
export type WorkflowSaved = Schemas['WorkflowSaved']
export type PromptList = Schemas['PromptList']
export type WorkspaceInfo = Schemas['WorkspaceInfo']
export type WorkspaceList = Schemas['WorkspaceList']
export type WorkspaceChanges = Schemas['WorkspaceChanges']
export type ChangeCounts = Schemas['ChangeCounts']
export type WorkspaceDeleted = Schemas['WorkspaceDeleted']
export type NoteSet = Schemas['NoteSet']
export type NoteRead = Schemas['NoteRead']
export type StarSet = Schemas['StarSet']
export type Deleted = Schemas['Deleted']
export type WorkflowDeleted = Schemas['WorkflowDeleted']
export type PromptSaved = Schemas['PromptSaved']
export type EnhancerPresets = Schemas['EnhancerPresets']
export type LoraList = Schemas['LoraList']
export type LoraRow = Schemas['LoraRow']
export type GalleryFile = Schemas['GalleryFile']
export type GalleryList = Schemas['GalleryList']
export type GalleryMetadata = Schemas['GalleryMetadata']
export type GalleryInput = Schemas['GalleryInput']
// A take of a request item that published a gallery file
export type GalleryLinkedJob = Schemas['GalleryLinkedJob']
// One finding, as GET /api/gallery/{name}/metadata (`findings`) and
// /assess send it (dw/assessment_rules.py `finding()`)
export type LevelFinding = Schemas['LevelFinding']
/** A file's role in its run, as the server states it. */
export type OutputRole = NonNullable<GalleryFile['role']>
export type OutputDeleted = Schemas['OutputDeleted']
export type AssetFile = Schemas['AssetFile']
/** The kinds of file the gallery and the asset library list */
export type OutputKind = GalleryFile['kind']
export type AssetKind = AssetFile['kind']
export type AssetLibrary = Schemas['AssetLibraryRoot']
export type ShadowedAsset = Schemas['ShadowedAsset']
export type AssetList = Schemas['AssetList']
export type Uploaded = Schemas['Uploaded']
export type Kept = Schemas['Kept']
export type AssetDeleted = Schemas['AssetDeleted']
// The requests ledger (dw/ledger.py): a workspace's list, one request's
// derived view, and its parts
export type LedgerList = Schemas['LedgerList']
export type LedgerSummary = Schemas['LedgerSummary']
export type LedgerRequest = Schemas['LedgerRequest']
export type LedgerDeleted = Schemas['LedgerDeleted']
export type LedgerItem = Schemas['LedgerItem']
export type LedgerReview = Schemas['LedgerReview']
export type LedgerAttempt = Schemas['LedgerAttempt']
export type LedgerOutput = Schemas['LedgerOutput']
export type LedgerNote = Schemas['LedgerNote']
export type LedgerNoteRef = Schemas['LedgerNoteRef']
// The request-notes routes: a new note, a new text, what they answer, and
// whether a notes channel is polling the workspace
export type NoteAdd = Schemas['NoteAdd']
export type NoteEdit = Schemas['NoteEdit']
export type NoteWritten = Schemas['NoteWritten']
export type NoteDeleted = Schemas['NoteDeleted']
export type NotesChannelState = Schemas['NotesChannelState']
export type LedgerHistoryEntry = Schemas['LedgerHistoryEntry']
export type LedgerItemStatus = LedgerItem['status']
export type PipelineParameter = Schemas['PipelineParameter']
export type PipelineDescription = Schemas['PipelineDescription']
export type PipelineNames = Schemas['PipelineNames']
export type TaskList = Schemas['TaskList']
export type BuiltinWorkflows = Schemas['BuiltinWorkflows']
export type ClassList = Schemas['ClassList']

export interface JobEvent {
  seq: number
  event: string
  [key: string]: unknown
}

/** GET /api/jobs/{id}/event-log (dw/server/routes/jobs.py, job_event_log).
 * The route has no response model, so this is a local narrowing; a later
 * `response_model` replaces it. `truncated` means this page was cut at
 * `limit`; `note` says what a finished job's history dropped, or null. */
export interface EventLogReply {
  id: string
  status: string | null
  events: JobEvent[]
  last_seq: number
  truncated: boolean
  note: string | null
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

/** The nodes of a workflow document (dw/workflow_schema.json), as the editor
 * reads them. Hand-written: the server publishes no TypeScript for them. A
 * file on disk may omit what the schema requires and still open, so a block
 * the UI does not itself create is optional - a read or a write through it
 * has to say what happens when it is missing. Any value may be a `variable:`
 * reference, and a key not named here is kept as `unknown`. */
export type StepArguments = Record<string, unknown>

export interface PipelineConfiguration {
  component_type?: string
  offload?: string
  attention_backend?: string
  prompt_weighting?: unknown
  cache?: string | CacheBlock
  [key: string]: unknown
}

export interface CacheBlock {
  type?: string
  threshold?: number | string
  [key: string]: unknown
}

export interface FromPretrainedArguments {
  model_name?: string
  torch_dtype?: string
  [key: string]: unknown
}

export interface QuantizationBlock {
  configuration?: { config_type?: string; [key: string]: unknown }
  arguments?: StepArguments
}

export interface PipelineComponent {
  configuration?: { component_type?: string; [key: string]: unknown }
  from_pretrained_arguments?: FromPretrainedArguments
  quantization_config?: QuantizationBlock
  group_offload?: {
    offload_type?: string
    use_stream?: boolean
    [key: string]: unknown
  }
  [key: string]: unknown
}

export interface SchedulerBlock {
  configuration?: { scheduler_type?: string; [key: string]: unknown }
  from_config_args?: StepArguments
  shift?: number | string
}

export interface LoraEntry {
  model_name?: string
  weight_name?: string
  revision?: string
  scale?: number | string
  [key: string]: unknown
}

export interface PipelineBlock {
  configuration?: PipelineConfiguration
  from_pretrained_arguments?: FromPretrainedArguments
  arguments?: StepArguments
  scheduler?: SchedulerBlock
  audio_scheduler?: SchedulerBlock
  loras?: LoraEntry[]
  chain?: Record<string, unknown>
  /** Every other key: an option block, or a component by its slot name */
  [key: string]: unknown
}

export interface TaskBlock {
  command?: string
  arguments?: StepArguments
  inputs?: unknown
  [key: string]: unknown
}

export interface SubWorkflowBlock {
  path?: string
  arguments?: StepArguments
}

export interface PipelineReferenceBlock {
  reference_name?: string
  arguments?: StepArguments
  chain?: Record<string, unknown>
}

export interface StepResult {
  content_type?: string
  save?: unknown
  file_base_name?: string
  subfolder?: string
  role?: string
  [key: string]: unknown
}

export interface WorkflowStep {
  name: string
  seed?: unknown
  for_each?: unknown
  pipeline?: PipelineBlock
  task?: TaskBlock
  workflow?: SubWorkflowBlock
  pipeline_reference?: PipelineReferenceBlock
  result?: StepResult
  [key: string]: unknown
}

export interface WorkflowDefinition {
  /** Required by the schema; a draft or hand-written file may still lack it */
  id?: string
  description?: string
  seed?: unknown
  variables?: Record<string, unknown>
  steps?: WorkflowStep[]
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
  /** The ETag of the bytes read, for a save's `If-Match`; null without one */
  etag: string | null
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
  /** The ETag of the bytes read, for a save's `If-Match`; null without one */
  etag: string | null
}

/** An asset a nearer library hides: same shape as `AssetFile` except there
 * is no `url` - that URL would serve the shadowing file, not this one - and
 * `shadowed_by` names the origin that won. */

/** GET /api/gallery/{name}/assess, the default (no `probe`) answer
 * (dw/server/assess.py). The route has no response model, so this is a
 * local narrowing; a later `response_model` replaces it. `at` is seconds
 * into the file, or null for a whole-file finding. */
export interface AssessReply {
  name: string
  source: 'output' | 'asset'
  kind: string | null
  shots_source?: string
  findings: Array<LevelFinding & { at: number | null }>
  rules_applied: string[]
  rules_skipped?: Array<{ probe: string; [key: string]: unknown }>
  /** Each probe the file cannot feed, with why: {probe: reason} */
  not_applicable: Record<string, string>
}
