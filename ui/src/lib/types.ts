export interface JobSummary {
  id: string
  workflow: string
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'
  created_at: number
  started_at: number | null
  finished_at: number | null
}

export interface ManifestEntry {
  step: string
  files: string[]
}

export interface JobDetail extends JobSummary {
  arguments: Record<string, unknown>
  warnings: string[]
  manifest: ManifestEntry[]
  error: string | null
  traceback: string | null
  event_count: number
}

export interface JobEvent {
  seq: number
  event: string
  [key: string]: unknown
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

export interface WorkflowDefinition {
  id: string
  variables?: Record<string, unknown>
  steps?: Array<Record<string, unknown>>
  [key: string]: unknown
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
  warnings: string[]
}

export interface GalleryFile {
  name: string
  url: string
  kind: 'image' | 'video' | 'audio'
  size: number
  mtime: number
  label: string
}
