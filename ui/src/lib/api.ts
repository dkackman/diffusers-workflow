import type {
  DiffusersStatus,
  JobDetail,
  ModelCache,
  ModelDownload,
  JobEvent,
  JobSummary,
  GalleryFile,
  MemoryInfo,
  PipelineDescription,
  ValidationResult,
  WorkflowDefinition,
} from './types'

/** Encode a workflow name for a URL, keeping its folder separators. */
const encodePath = (name: string) =>
  name.split('/').map(encodeURIComponent).join('/')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    let detail = response.statusText
    try {
      detail = (await response.json()).detail ?? detail
    } catch {
      /* not json */
    }
    throw new Error(detail)
  }
  return response.json()
}

export const api = {
  listWorkflows: () =>
    request<{
      workflow_dir: string
      workflows: string[]
      details: Record<
        string,
        { kinds: string[]; variables: number; description: string }
      >
    }>('/api/workflows'),
  getWorkflow: (name: string) =>
    request<WorkflowDefinition>(`/api/workflows/${encodePath(name)}`),
  listJobs: () => request<{ jobs: JobSummary[] }>('/api/jobs'),
  getJob: (id: string) => request<JobDetail>(`/api/jobs/${id}`),
  rerunJob: (id: string) =>
    request<JobDetail>(`/api/jobs/${id}/rerun`, { method: 'POST' }),
  listTasks: () =>
    request<{
      commands: string[]
      image_processors: string[]
      video_processors: string[]
    }>('/api/tasks'),
  describeTask: (command: string) =>
    request<PipelineDescription>(`/api/tasks/${encodeURIComponent(command)}`),
  moveJob: (id: string, direction: 'up' | 'down' | 'front' | 'back') =>
    request<{ id: string; queue: string[] }>(`/api/jobs/${id}/move`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ direction }),
    }),
  cancelJob: (id: string) =>
    request<{ id: string; status: string }>(`/api/jobs/${id}/cancel`, {
      method: 'POST',
    }),
  submitJob: (body: {
    workflow_path?: string
    workflow?: WorkflowDefinition
    arguments?: Record<string, unknown>
    base_dir?: string
  }) =>
    request<JobDetail>('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  memory: () => request<MemoryInfo>('/api/memory'),
  listModels: () => request<ModelCache>('/api/models'),
  startDownload: (repoId: string) =>
    request<ModelDownload>('/api/models/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_id: repoId }),
    }),
  listDownloads: () =>
    request<{ downloads: ModelDownload[] }>('/api/models/downloads'),
  cancelDownload: (id: string) =>
    request<ModelDownload>(`/api/models/downloads/${id}/cancel`, {
      method: 'POST',
    }),
  deleteModel: (repo: string) =>
    request<{ repo_id: string; deleted: boolean; freed: number }>(
      `/api/models?repo=${encodeURIComponent(repo)}`,
      { method: 'DELETE' },
    ),
  diffusersStatus: () => request<DiffusersStatus>('/api/system/diffusers'),
  updateDiffusers: () =>
    request<DiffusersStatus>('/api/system/diffusers/update', {
      method: 'POST',
    }),
  health: () =>
    request<{
      status: string
      worker_alive: boolean
      current_job: string | null
    }>('/api/health'),
  gallery: (limit = 200) =>
    request<{ files: GalleryFile[]; total: number }>(
      `/api/gallery?limit=${limit}`,
    ),
  galleryMetadata: (name: string) =>
    request<{
      name: string
      metadata: Record<string, unknown> | null
      job: { id: string; status: string } | null
    }>(`/api/gallery/${encodeURIComponent(name)}/metadata`),
  deleteOutput: (name: string) =>
    request<{ name: string; deleted: boolean }>(
      `/api/gallery/${encodeURIComponent(name)}`,
      { method: 'DELETE' },
    ),
  listPipelines: () => request<{ pipelines: string[] }>('/api/pipelines'),
  describePipeline: (name: string) =>
    request<PipelineDescription>(`/api/pipelines/${name}`),
  listClasses: (kind: string) =>
    request<{ kind: string; classes: string[] }>(`/api/classes?kind=${kind}`),
  describeClass: (name: string, target: 'call' | 'init' | 'load') =>
    request<PipelineDescription>(
      `/api/classes/${encodeURIComponent(name)}?target=${target}`,
    ),
  getSchema: () => request<Record<string, unknown>>('/api/schema'),
  validate: (workflow: WorkflowDefinition) =>
    request<ValidationResult>('/api/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workflow }),
    }),
  deleteWorkflow: (name: string) =>
    request<{ name: string; deleted: boolean }>(
      `/api/workflows/${encodePath(name)}`,
      { method: 'DELETE' },
    ),
  saveWorkflow: (name: string, workflow: WorkflowDefinition) =>
    request<{ name: string; path: string; warnings: string[] }>(
      `/api/workflows/${encodePath(name)}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow }),
      },
    ),
}

const TERMINAL_STATUSES = ['succeeded', 'failed', 'cancelled']

/** Stream a job's events; returns a stop function. The stream closes itself
 * when a terminal job_status arrives; transient errors are left alone so
 * EventSource reconnects and resumes losslessly via Last-Event-ID. */
export function streamJobEvents(
  jobId: string,
  after: number,
  onEvent: (event: JobEvent) => void,
  onEnd: () => void,
): () => void {
  const source = new EventSource(`/api/jobs/${jobId}/events?after=${after}`)
  source.onmessage = (message) => {
    const event: JobEvent = JSON.parse(message.data)
    onEvent(event)
    if (
      event.event === 'job_status' &&
      TERMINAL_STATUSES.includes(event.status as string)
    ) {
      source.close()
      onEnd()
    }
  }
  // No onerror handling: a dropped connection is EventSource's own job to
  // repair. Closing here froze live progress on any transient hiccup.
  return () => source.close()
}
