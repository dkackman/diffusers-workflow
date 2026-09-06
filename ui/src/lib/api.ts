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
  ServerInfo,
  ValidationResult,
  WorkflowDefinition,
  WorkflowWithOrigin,
} from './types'
import { getApiToken } from './token'
import { DEFAULT_WORKSPACE, workspace } from './workspace.svelte'

/** Encode a workflow name for a URL, keeping its folder separators. */
const encodePath = (name: string) =>
  name.split('/').map(encodeURIComponent).join('/')

/** Append one query parameter to a URL, keeping whatever query it already
 * has. Shared by every place that tacks a selector onto a path - the
 * workspace scope, the download token - so there is one rule for `?` vs
 * `&` instead of a hand-rolled check at each call site. */
function appendQuery(url: string, key: string, value: string): string {
  const separator = url.includes('?') ? '&' : '?'
  return `${url}${separator}${key}=${encodeURIComponent(value)}`
}

/** Append the workspace selector to a path, keeping any query it has. The
 * server defaults to this one when no selector is sent, so 'default' sends
 * nothing and the request looks exactly as it did before workspaces
 * existed. Routes that are not workspace-scoped (prompts, models, system)
 * ignore an unknown query parameter, which is what lets this live in one
 * place instead of being threaded through every call site. */
function scoped(path: string): string {
  if (workspace.current === DEFAULT_WORKSPACE) return path
  return appendQuery(path, 'workspace', workspace.current)
}

/** Append the configured API token as a query parameter. Only for the
 * routes a browser loads without being able to set headers - EventSource,
 * <img> tags and <a download> navigations - which the server accepts it
 * on; see docs/SERVER.md. */
function withToken(url: string): string {
  const scopedUrl = scoped(url)
  const token = getApiToken()
  return token ? appendQuery(scopedUrl, 'token', token) : scopedUrl
}

/** The URL an output file is served from. Jobs report files by their name
 * relative to the output directory - a workflow under a subfolder writes
 * to '<sub>/<file>' - so the whole relative path is kept. A job recorded
 * before that change carries an absolute path, for which the basename is
 * the best available guess. `version` busts the browser cache: two runs of
 * one workflow write the same file names. `workspace`, when given, names
 * the job's own workspace and wins over whatever is currently selected in
 * the picker - a job page must load its files from where they were written,
 * not from wherever the user has since navigated to. */
export function outputUrl(
  path: string,
  version?: string,
  workspace?: string,
): string {
  const name = path.startsWith('/') ? (path.split('/').pop() ?? '') : path
  const url = `/outputs/${encodePath(name)}`
  const versioned =
    version === undefined ? url : appendQuery(url, 'v', version)
  if (workspace === undefined) return scoped(versioned)
  return workspace === DEFAULT_WORKSPACE
    ? versioned
    : appendQuery(versioned, 'workspace', workspace)
}

const BYTES_PER_MB = 1024 * 1024

/** The per-folder file counts a workspace delete answers with, as one line.
 * Folders holding nothing are left out - the point of the count is what
 * would actually be lost. */
function describeContents(contents: unknown): string {
  if (!contents || typeof contents !== 'object') return ''
  const parts: string[] = []
  for (const [folder, value] of Object.entries(
    contents as Record<string, { files?: number; bytes?: number }>,
  )) {
    const files = Number(value?.files ?? 0)
    if (!files) continue
    const bytes = Number(value?.bytes ?? 0)
    const size =
      bytes >= BYTES_PER_MB ? ` (${(bytes / BYTES_PER_MB).toFixed(1)} MB)` : ''
    parts.push(`${folder}: ${files} file${files === 1 ? '' : 's'}${size}`)
  }
  return parts.length ? parts.join(', ') : 'nothing'
}

/** The message inside an error response's `detail`. Most routes answer with
 * a plain string, but the ones that have to say what they would do send an
 * object instead - the workspace delete's `{message, contents}`, the
 * not-a-workspace refusal's `{message, entries}` - and handing that straight
 * to `new Error` shows the user '[object Object]' rather than the very
 * numbers the confirmation exists to present. FastAPI's 422 list of
 * validation errors gets the same treatment. */
export function errorDetail(payload: unknown, fallback: string): string {
  const detail = (payload as { detail?: unknown } | null)?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    const messages = detail
      .map((entry) =>
        entry && typeof entry === 'object'
          ? String((entry as { msg?: unknown }).msg ?? JSON.stringify(entry))
          : String(entry),
      )
      .filter(Boolean)
    return messages.length ? messages.join('. ') : fallback
  }
  if (detail && typeof detail === 'object') {
    const record = detail as Record<string, unknown>
    const message =
      typeof record.message === 'string' ? record.message : fallback
    const contents = describeContents(record.contents)
    if (contents) return `${message}\n\n${contents}`
    if (Array.isArray(record.entries) && record.entries.length) {
      return `${message}\n\n${record.entries.join(', ')}`
    }
    return message
  }
  return fallback
}

/** A JSON response together with its headers, for the rare endpoint whose
 * result depends on both - `getWorkflow` reads its origin/writable from
 * headers rather than the body. */
async function fetchJson<T>(
  path: string,
  init?: RequestInit,
  options?: { scope?: boolean },
): Promise<{ body: T; response: Response }> {
  const token = getApiToken()
  const headers = new Headers(init?.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const url = options?.scope === false ? path : scoped(path)
  const response = await fetch(url, { ...init, headers })
  if (!response.ok) {
    let detail = response.statusText
    try {
      detail = errorDetail(await response.json(), detail)
    } catch {
      /* not json */
    }
    throw new Error(detail)
  }
  return { body: await response.json(), response }
}

async function request<T>(
  path: string,
  init?: RequestInit,
  options?: { scope?: boolean },
): Promise<T> {
  return (await fetchJson<T>(path, init, options)).body
}

/** Fetch a file-response endpoint and hand the body to the browser as a
 * save, using the name the server put in Content-Disposition. Separate
 * from `request` because the body is a file, not JSON, and separate from
 * the `<a download>` URLs because the request needs a method and a body. */
async function downloadResponse(
  path: string,
  init: RequestInit,
): Promise<void> {
  const token = getApiToken()
  const headers = new Headers(init.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const response = await fetch(scoped(path), { ...init, headers })
  if (!response.ok) {
    let detail = response.statusText
    try {
      detail = errorDetail(await response.json(), detail)
    } catch {
      /* not json */
    }
    throw new Error(detail)
  }
  const disposition = response.headers.get('content-disposition') ?? ''
  const filename = /filename="?([^"]+)"?/.exec(disposition)?.[1] ?? 'download'
  const url = URL.createObjectURL(await response.blob())
  try {
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = filename
    anchor.click()
  } finally {
    URL.revokeObjectURL(url)
  }
}

export const api = {
  listWorkflows: () =>
    request<{
      /** The writable directory - where a save lands, whatever source a
       * workflow was read from. */
      workflow_dir: string
      /** The search path, writable root first. */
      sources?: { root: string; origin: string; writable: boolean }[]
      workflows: string[]
      details: Record<
        string,
        {
          kinds: string[]
          steps?: number
          variables: number
          description: string
          prompt_refs?: string[]
          /** Which source it came from: 'workspace', 'examples', 'builtin'. */
          origin?: string
          /** False for a read-only source: offer save-a-copy, not delete. */
          writable?: boolean
        }
      >
    }>('/api/workflows'),
  /** The workflow plus where it came from, read off the response headers
   * rather than a separate `listWorkflows` lookup. */
  getWorkflow: (name: string) =>
    fetchJson<WorkflowDefinition>(`/api/workflows/${encodePath(name)}`).then(
      ({ body, response }): WorkflowWithOrigin => ({
        ...body,
        origin: response.headers.get('X-Workflow-Origin') ?? '',
        writable: response.headers.get('X-Workflow-Writable') !== 'false',
      }),
    ),
  // Unscoped: the jobs list spans every workspace on purpose, with its own
  // filter dropdown rather than following wherever the picker points.
  // Omitting `workspace` returns jobs from all of them.
  listJobs: (workspace?: string) =>
    request<{ jobs: JobSummary[] }>(
      workspace ? `/api/jobs?workspace=${encodeURIComponent(workspace)}` : '/api/jobs',
      undefined,
      { scope: false },
    ),
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
  server: () => request<ServerInfo>('/api/server'),
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
  /** Download a multi-file gallery selection as one zip. The browser
   * cannot zip on its own and throttles a burst of single downloads, so
   * the server bundles the selection and this saves the response. */
  archiveOutputs: (names: string[]) =>
    downloadResponse('/api/gallery/archive', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ names }),
    }),
  /** Save a browser-picked file server-side and get back the path a
   * workflow's image/video argument can reference. The body is the raw
   * file bytes - no multipart form needed for a single file. */
  uploadMedia: (file: File) =>
    request<{ path: string; url: string }>(
      `/api/uploads?filename=${encodeURIComponent(file.name)}`,
      { method: 'POST', body: file },
    ),
  listWorkspaces: () =>
    request<{
      workspace_root: string | null
      default: string
      workspaces: {
        name: string
        default: boolean
        workflows: string
        assets: string | null
        outputs: string
        prompts: string | null
      }[]
    }>('/api/workspaces'),
  createWorkspace: (name: string) =>
    request<{ name: string }>('/api/workspaces', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }),
  /** Delete a workspace and everything in it. The server refuses without
   * `acknowledged`, answering with what it would remove - so the caller can
   * show that before asking again. */
  deleteWorkspace: (name: string, acknowledged = false) =>
    request<{ name: string; deleted: boolean }>(
      `/api/workspaces/${encodeURIComponent(name)}?acknowledged=${acknowledged}`,
      { method: 'DELETE' },
    ),
  /** Keep a generated file as an input asset under a stable name. The copy
   * happens on the server, inside the workspace - nothing is downloaded and
   * re-uploaded to reuse a render. */
  keepOutput: (name: string, assetName?: string, overwrite = false) =>
    request<{ reference: string; name: string; linked: boolean }>(
      '/api/assets/keep',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          asset_name: assetName ?? null,
          overwrite,
        }),
      },
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
 * comes back, since the manifest only names files. `workspace` names the
 * job's own workspace, for the same reason `outputUrl` takes one: the file
 * has to be read from where the job wrote it, not from whatever the picker
 * says now. */
export async function fetchOutputText(
  path: string,
  workspace?: string,
): Promise<string> {
  const name = path.split('/').pop() ?? ''
  const response = await fetch(outputUrl(path, undefined, workspace))
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
