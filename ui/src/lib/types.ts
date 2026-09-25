export interface JobSummary {
  id: string
  workflow: string
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'
  created_at: number
  started_at: number | null
  finished_at: number | null
  historical?: boolean
  queue_position?: number
  /** The workspace this job ran in - 'default' for the default one. */
  workspace: string
  /** Which form of cost acknowledgement queued the job: none (the web UI
   * and any caller that sent nothing), a bare boolean, or one bound to the
   * plan a validate answered with. Absent on rows from older servers. */
  acknowledged?: 'none' | 'boolean' | 'bound'
  /** The run this job opened - null until it opens one, and for a job
   * recorded before runs were tracked. */
  run_id?: string | null
  /** That run's ordinal among the workflow's runs - the `v4` the gallery
   * shows for its files. Null until the run opens, and for older rows. */
  run_version?: number | null
}

export interface ManifestEntry {
  step: string
  files: string[]
  /** The step was served from the step cache: these files are an earlier
   * run's, republished, and nothing was generated for them this time. */
  reused?: boolean
  /** The in-run subfolder the step's `result.subfolder` chose - `final`,
   * `intermediate`, any relative path - `''` when it chose none. Absent
   * only on a job recorded before the field existed. */
  subfolder?: string
}

export interface JobDetail extends JobSummary {
  arguments: Record<string, unknown>
  warnings: string[]
  manifest: ManifestEntry[]
  error: string | null
  traceback: string | null
  event_count: number
  /** The plan the caller bound its acknowledgement to, when it did. */
  acknowledged_cost?: AcknowledgedCost | null
}

export interface AcknowledgedCost {
  fingerprint: string
  minutes?: number | null
  downloads?: string[]
}

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

export interface HealthInfo {
  status: string
  version?: string
  hostname?: string
  device?: string
  mcp?: boolean
  worker_alive: boolean
  current_job: string | null
  queued?: number
}

export interface ServerAddress {
  address: string
  family: string
  interface: string | null
}

export interface ServerInfo {
  hostname: string
  version: string
  device: string
  bind_host: string
  port: number
  wildcard_bind: boolean
  auth_required: boolean
  mcp: { mounted: boolean; path: string }
  addresses: ServerAddress[]
  directories: {
    /** The workspace the folders below are folders of, when the server
     * resolved one; an individually overridden folder still reports its
     * own path. */
    workspace: string | null
    workflows: string
    outputs: string
    prompts: string | null
    assets: string | null
  }
}

export interface MemoryInfo {
  live: boolean
  info: {
    gpu_available?: boolean
    gpu_device_name?: string
    gpu_memory_allocated_mb?: number
    gpu_memory_total_mb?: number
    run_count?: number
  } | null
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

/** One measured run. `name` is the accelerator for a person ('RTX 4090')
 * and is optional - only `device`, `vram_gb` and `minutes` are required. */
export interface WorkflowCost {
  device: string
  name?: string
  vram_gb: number
  minutes: number
}

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

export interface PromptDetail {
  description: string
  intended_model: string
  tags: string[]
  text: string
}

export interface EnhancerPreset {
  key: string
  label: string
  default_model: string
  models: string[]
  intended_models: string[]
  placeholder: string
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

export interface ValidationResult {
  valid: boolean
  error: string | null
  /** Every schema violation with its JSON path; empty when valid. */
  errors: { path: string | null; message: string }[]
  warnings: string[]
  /** What the run will execute for the definition validated - on a valid
   * answer; null when the server could not build it, absent from older
   * servers and from an invalid answer. */
  plan?: Plan | null
}

/** A validate answer's plan: the work a run will do, priced from the
 * workflow's own cost block, with the weights this box lacks named. */
export interface Plan {
  fingerprint: string
  steps: number
  list_entries: Record<string, number>
  /** How many steps the worker's step cache would serve; null when the
   * worker was busy or did not answer. */
  cached_steps: number | null
  /** The steps that will not run because nothing reads their result and
   * they save no file - already excluded from `steps` (#122). */
  elided_steps: { step: string; reason: string }[]
  downloads_required: { repo: string | null; url?: string; gb: number | null }[]
  estimate: {
    minutes: number | null
    basis: 'per_entry' | 'catalog' | 'derived' | 'other_device' | 'unknown'
    device: string
    measured_on: string | null
    partial: boolean
    /** What contributed nothing to `minutes` when `partial` is true - the
     * workflow's own id when its own steps went unpriced, else the path of
     * each composed child with no cost block. Empty when `partial` is false. */
    unpriced: string[]
  }
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

/** One root on the asset search path - what `asset_dirs` names, plus the
 * origin and writability a client needs to explain why a delete can reach
 * one root and not another. */
export interface AssetLibrary {
  origin: AssetFile['origin']
  dir: string
  writable: boolean
}

/** An asset a nearer library hides: same shape as `AssetFile` except there
 * is no `url` - that URL would serve the shadowing file, not this one - and
 * `shadowed_by` names the origin that won. */
export type ShadowedAsset = Omit<AssetFile, 'url'> & {
  shadowed_by: AssetFile['origin']
}

export interface ModelRevision {
  commit_hash: string
  size_on_disk: number
  refs: string[]
  last_modified: number | null
}

export interface ModelRepo {
  repo_id: string
  repo_type: string
  size_on_disk: number
  nb_files: number
  last_accessed: number | null
  last_modified: number | null
  revisions: ModelRevision[]
}

export interface ModelCache {
  cache_dir: string
  size_on_disk: number
  repos: ModelRepo[]
  warnings: string[]
  disk_free: number | null
  disk_total: number | null
}

export interface ModelDownload {
  id: string
  repo_id: string
  status: 'downloading' | 'completed' | 'cancelled' | 'failed'
  downloaded: number
  total: number | null
  error: string | null
  started_at: number
  finished_at: number | null
}

export interface DiffusersStatus {
  status: 'idle' | 'running' | 'succeeded' | 'failed'
  error: string | null
  log: string | null
  started_at: number | null
  finished_at: number | null
  version: string | null
  commit: string | null
}
