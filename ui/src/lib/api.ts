import type {
  DiffusersStatus,
  EnhancerPreset,
  JobDetail,
  ModelCache,
  ModelDownload,
  JobEvent,
  JobSummary,
  GalleryFile,
  HealthInfo,
  MemoryInfo,
  PipelineDescription,
  PromptDefinition,
  PromptDetail,
  ValidationResult,
  WorkflowDefinition,
} from './types'
import { getApiToken } from './token'

/** Encode a workflow name for a URL, keeping its folder separators. */
const encodePath = (name: string) =>
  name.split('/').map(encodeURIComponent).join('/')

/** Append the configured API token as a query parameter. Only for the
 * routes a browser loads without being able to set headers - EventSource,
 * <img> tags and <a download> navigations - which the server accepts it
 * on; see docs/SERVER.md. */
function withToken(url: string): string {
  const token = getApiToken()
  if (!token) return url
  const separator = url.includes('?') ? '&' : '?'
  return `${url}${separator}token=${encodeURIComponent(token)}`
}

/** The URL an output file is served from. Jobs report files by their name
 * relative to the output directory - a workflow under a subfolder writes
 * to '<sub>/<file>' - so the whole relative path is kept. A job recorded
 * before that change carries an absolute path, for which the basename is
 * the best available guess. `version` busts the browser cache: two runs of
 * one workflow write the same file names. */
export function outputUrl(path: string, version?: string): string {
  const name = path.startsWith('/') ? (path.split('/').pop() ?? '') : path
  const url = `/outputs/${encodePath(name)}`
  return version === undefined ? url : `${url}?v=${encodeURIComponent(version)}`
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getApiToken()
  const headers = new Headers(init?.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const response = await fetch(path, { ...init, headers })
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
        {
          kinds: string[]
          steps?: number
          variables: number
          description: string
          prompt_refs?: string[]
        }
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
  health: () => request<HealthInfo>('/api/health'),
  // Loads the whole gallery in one request, like listWorkflows/listPrompts -
  // the limit just needs to exceed any real output directory's file count
  gallery: () => request<{ files: GalleryFile[] }>('/api/gallery?limit=100000'),
  galleryMetadata: (name: string) =>
    request<{
      name: string
      metadata: Record<string, unknown> | null
      job: { id: string; status: string } | null
    }>(`/api/gallery/${encodePath(name)}/metadata`),
  galleryThumbnailUrl: (name: string) =>
    withToken(`/api/gallery/${encodePath(name)}/thumbnail`),
  deleteOutput: (name: string) =>
    request<{ name: string; deleted: boolean }>(
      `/api/gallery/${encodePath(name)}`,
      { method: 'DELETE' },
    ),
  outputDownloadUrl: (name: string) =>
    withToken(`/api/gallery/${encodePath(name)}/download`),
  /** Save a browser-picked file server-side and get back the path a
   * workflow's image/video argument can reference. The body is the raw
   * file bytes - no multipart form needed for a single file. */
  uploadMedia: (file: File) =>
    request<{ path: string; url: string }>(
      `/api/uploads?filename=${encodeURIComponent(file.name)}`,
      { method: 'POST', body: file },
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
  workflowDownloadUrl: (name: string) =>
    withToken(`/api/workflows/${encodePath(name)}/download`),
  saveWorkflow: (name: string, workflow: WorkflowDefinition) =>
    request<{ name: string; path: string; warnings: string[] }>(
      `/api/workflows/${encodePath(name)}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow }),
      },
    ),
  listPrompts: () =>
    request<{
      prompt_dir: string
      prompts: string[]
      details: Record<string, PromptDetail>
    }>('/api/prompts'),
  getPrompt: (name: string) =>
    request<PromptDefinition>(`/api/prompts/${encodePath(name)}`),
  savePrompt: (name: string, prompt: PromptDefinition) =>
    request<{ name: string; path: string }>(
      `/api/prompts/${encodePath(name)}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      },
    ),
  deletePrompt: (name: string) =>
    request<{ name: string; deleted: boolean }>(
      `/api/prompts/${encodePath(name)}`,
      { method: 'DELETE' },
    ),
  promptDownloadUrl: (name: string) =>
    withToken(`/api/prompts/${encodePath(name)}/download`),
  getPromptSchema: () => request<Record<string, unknown>>('/api/prompt-schema'),
  listEnhancers: () => request<{ presets: EnhancerPreset[] }>('/api/enhancers'),
  enhance: (body: {
    idea: string
    preset: string
    model_name?: string
    device?: string
  }) =>
    request<JobDetail>('/api/enhance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
}

/** Fetch the text of a saved output file - how an enhancement's result
 * comes back, since the manifest only names files. */
export async function fetchOutputText(path: string): Promise<string> {
  const name = path.split('/').pop() ?? ''
  const response = await fetch(outputUrl(path))
  if (!response.ok) throw new Error(`Could not read ${name}`)
  return response.text()
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
  // EventSource cannot set custom headers, so a configured token rides
  // along as a query parameter for this one route - see docs/SERVER.md.
  const source = new EventSource(
    withToken(`/api/jobs/${jobId}/events?after=${after}`),
  )
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
