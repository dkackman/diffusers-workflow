import type {
  JobDetail,
  JobEvent,
  JobSummary,
  GalleryFile,
  MemoryInfo,
  PipelineDescription,
  ValidationResult,
  WorkflowDefinition,
} from './types'

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
      details: Record<string, { kinds: string[]; variables: number }>
    }>('/api/workflows'),
  getWorkflow: (name: string) =>
    request<WorkflowDefinition>(`/api/workflows/${name}`),
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
  cancelJob: (id: string) =>
    request<{ id: string; status: string }>(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  submitJob: (body: {
    workflow_path?: string
    workflow?: WorkflowDefinition
    arguments?: Record<string, unknown>
  }) =>
    request<JobDetail>('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  memory: () => request<MemoryInfo>('/api/memory'),
  health: () =>
    request<{ status: string; worker_alive: boolean; current_job: string | null }>(
      '/api/health',
    ),
  gallery: (limit = 200) =>
    request<{ files: GalleryFile[]; total: number }>(`/api/gallery?limit=${limit}`),
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
    request<{ name: string; deleted: boolean }>(`/api/workflows/${name}`, {
      method: 'DELETE',
    }),
  saveWorkflow: (name: string, workflow: WorkflowDefinition) =>
    request<{ name: string; path: string; warnings: string[] }>(
      `/api/workflows/${name}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow }),
      },
    ),
}

/** Stream a job's events; returns a stop function. The server replays from
 * `after`, and EventSource reconnects carry Last-Event-ID automatically. */
export function streamJobEvents(
  jobId: string,
  after: number,
  onEvent: (event: JobEvent) => void,
  onEnd: () => void,
): () => void {
  const source = new EventSource(`/api/jobs/${jobId}/events?after=${after}`)
  source.onmessage = (message) => onEvent(JSON.parse(message.data))
  source.onerror = () => {
    // The stream closes when the job reaches a terminal state; EventSource
    // then fires error while trying to reconnect against a finished stream
    if (source.readyState === EventSource.CONNECTING) {
      source.close()
      onEnd()
    }
  }
  return () => source.close()
}
