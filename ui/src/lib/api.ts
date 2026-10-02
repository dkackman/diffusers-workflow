import type {
  AssetFile,
  AssetLibrary,
  ShadowedAsset,
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
  LibraryRoot,
  ShadowedEntry,
  ServerInfo,
  ValidationResult,
  WorkflowCost,
  WorkflowDefinition,
  WorkflowShape,
  WorkflowTrait,
  StoredPrompt,
  WorkflowWithOrigin,
} from './types'
import { getApiToken } from './token'
import { ApiError, errorDetail } from './apiErrors'
import {
  addToken,
  appendQuery,
  encodePath,
  outputUrl,
  scopeTo,
  scoped,
  withToken,
} from './apiUrls'

// Pages and their test mocks import these from here
export { ApiError, errorDetail, outputUrl }

/** A JSON response together with its headers, for the rare endpoint whose
 * result depends on both - `getWorkflow` reads its origin/writable from
 * headers rather than the body. */
/** A request with the bearer token, scoped to the picker's workspace
 * unless `scope` is false, failing with an ApiError that carries the
 * server's detail and status. Under every JSON call and file download. */
async function send(
  path: string,
  init: RequestInit = {},
  scope = true,
): Promise<Response> {
  const token = getApiToken()
  const headers = new Headers(init.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const response = await fetch(scope ? scoped(path) : path, {
    ...init,
    headers,
  })
  if (!response.ok) {
    let detail = response.statusText
    try {
      detail = errorDetail(await response.json(), detail)
    } catch {
      /* not json */
    }
    throw new ApiError(detail, response.status)
  }
  return response
}

async function fetchJson<T>(
  path: string,
  init?: RequestInit,
  options?: { scope?: boolean },
): Promise<{ body: T; response: Response }> {
  const response = await send(path, init, options?.scope !== false)
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
  const response = await send(path, init)
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

/** A bulk-download endpoint: POST the selection, save the zip that comes
 * back. The gallery and the asset library each have one. */
function archiveFrom(path: string) {
  return (names: string[]) =>
    downloadResponse(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ names }),
    })
}

export const api = {
  listWorkflows: () =>
    request<{
      workspace?: string
      /** The search path in order; the writable workspace root is where a
       * save lands, whatever library a workflow was read from. */
      libraries: LibraryRoot[]
      shadowed: ShadowedEntry[]
      workflows: string[]
      details: Record<
        string,
        {
          kinds: string[]
          steps?: number
          variables: number
          description: string
          /** For a model config: the template it is a tuned instance of.
           * Absent from an older server, and from every template. */
          configures?: string
          prompt_refs?: string[]
          /** What the workflow makes, derived by the server from the
           * definition (dw/server/catalog_shape.py). Absent from an older
           * server. */
          shape?: WorkflowShape
          /** Sorted, independent facts about how the output is made or what
           * it needs. */
          traits?: WorkflowTrait[]
          /** The description's first sentence, clipped - what a card shows. */
          summary?: string
          /** Measured runs, one per device the maintainer measured on. Null
           * (or absent) means unknown - never derived. */
          cost?: WorkflowCost[] | null
          /** Which source it came from: 'workspace', 'examples', 'builtin'. */
          origin: string
          /** False for a read-only source: offer save-a-copy, not delete. */
          writable: boolean
        }
      >
    }>('/api/workflows'),
  /** The workflow plus where it came from, read off the response headers
   * rather than a separate `listWorkflows` lookup. Beside the definition,
   * the way `getPrompt` keeps a prompt's: the object the caller holds is
   * what validate, save and run send back, and the schema refuses unknown
   * root keys, so the transport metadata must not ride inside it. */
  getWorkflow: (name: string) =>
    fetchJson<WorkflowDefinition>(`/api/workflows/${encodePath(name)}`).then(
      ({ body, response }): WorkflowWithOrigin => ({
        definition: body,
        origin: response.headers.get('X-Workflow-Origin') ?? '',
        writable: response.headers.get('X-Workflow-Writable') !== 'false',
      }),
    ),
  // Unscoped on purpose: `workspace` is explicit so Status can span every
  // workspace and a workspace's Jobs page can name its own. `status` is one
  // state or a comma-separated set ('succeeded,failed,cancelled'); `limit`
  // keeps the newest N of what matched.
  listJobs: (workspace?: string, limit?: number, status?: string) => {
    const query = new URLSearchParams()
    if (workspace) query.set('workspace', workspace)
    if (status) query.set('status', status)
    if (limit) query.set('limit', String(limit))
    const qs = query.toString()
    return request<{ jobs: JobSummary[]; total?: number }>(
      qs ? `/api/jobs?${qs}` : '/api/jobs',
      undefined,
      { scope: false },
    )
  },
  getJob: (id: string) => request<JobDetail>(`/api/jobs/${id}`),
  /** The definition a job ran, for the job page's read-only flow view.
   * `realized` true means `definition` is the realized copy the run itself
   * wrote - every mutable input pinned - rather than the definition as
   * submitted. 404s when the job named a workflow file that is no longer
   * readable. */
  getJobWorkflow: (id: string) =>
    request<{
      id: string
      definition: Record<string, any>
      realized: boolean
      /** The variable a new-seed rerun would draw into, null when the
       * workflow has none - the cue for whether to offer that at all. */
      seed_variable: string | null
    }>(`/api/jobs/${id}/workflow`),
  /** Queue the job again. `newSeed` draws a fresh seed into the workflow's
   * seed variable; without it the arguments repeat exactly, which the step
   * cache serves from the earlier run rather than generating anything. */
  rerunJob: (id: string, newSeed = false) =>
    request<JobDetail>(`/api/jobs/${id}/rerun`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_seed: newSeed }),
    }),
  listTasks: () =>
    request<{
      commands: string[]
      image_processors: string[]
      video_processors: string[]
      assessment: string[]
    }>('/api/tasks'),
  describeTask: (command: string) =>
    request<PipelineDescription>(`/api/tasks/${encodeURIComponent(command)}`),
  moveJob: (id: string, direction: 'up' | 'down' | 'front' | 'back') =>
    request<{ id: string; queue: string[] }>(`/api/jobs/${id}/move`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ direction }),
    }),
  /** Gather a finished job into '<workspace>/exports/<job id>/' on the
   * server - the realized workflow, manifest, job row, the media it used
   * and made - and get back the zip URL it is fetched from. Scoped to the
   * job's own workspace rather than the picker's, as the job page's media
   * is: the export belongs beside the run, wherever the user has since
   * navigated. 409 (an `ApiError`) when the export exists and `overwrite`
   * was not asked for. */
  exportJob: (id: string, workspace: string, overwrite = false) =>
    request<{
      directory: string
      zip_url: string
      files: { path: string; bytes: number }[]
      total_bytes: number
      missing: string[]
    }>(
      appendQuery(
        scopeTo(`/api/jobs/${id}/export`, workspace),
        'overwrite',
        overwrite ? 'true' : 'false',
      ),
      { method: 'POST' },
      { scope: false },
    ),
  /** The `zip_url` an export answered with, ready for an <a download>. The
   * server already put the job's workspace selector on it, so only the
   * token is added - `withToken` would scope it a second time. */
  exportZipUrl: (zipUrl: string) => addToken(zipUrl),
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
  // `workspace`, when given, names the file's own workspace and wins over
  // whatever is currently selected in the picker - mirrors `outputUrl`, since
  // a job page must read its own files from where they were written
  galleryMetadata: (name: string, workspace?: string) =>
    request<{
      name: string
      metadata: Record<string, unknown> | null
      job: { id: string; status: string } | null
    }>(
      workspace === undefined
        ? `/api/gallery/${encodePath(name)}/metadata`
        : scopeTo(`/api/gallery/${encodePath(name)}/metadata`, workspace),
      undefined,
      { scope: workspace === undefined },
    ),
  galleryThumbnailUrl: (name: string) =>
    withToken(`/api/gallery/${encodePath(name)}/thumbnail`),
  deleteOutput: (name: string) =>
    request<{ name: string; deleted: boolean }>(
      `/api/gallery/${encodePath(name)}`,
      { method: 'DELETE' },
    ),
  outputDownloadUrl: (name: string, workspace?: string) =>
    withToken(`/api/gallery/${encodePath(name)}/download`, workspace),
  /** Download a multi-file gallery selection as one zip. The browser
   * cannot zip on its own and throttles a burst of single downloads, so
   * the server bundles the selection and this saves the response. */
  archiveOutputs: archiveFrom('/api/gallery/archive'),
  /** Save a browser-picked file server-side and get back the path a
   * workflow's image/video argument can reference. The body is the raw
   * file bytes - no multipart form needed for a single file.
   *
   * `assetName` stores it under a name of the caller's choosing rather
   * than the random one a browser upload gets, and `shared` puts it in the
   * library every workspace under this root shares. */
  uploadMedia: (file: File, assetName?: string, shared = false) =>
    request<{ url: string; reference?: string }>(
      `/api/uploads?filename=${encodeURIComponent(file.name)}` +
        (assetName ? `&asset_name=${encodeURIComponent(assetName)}` : '') +
        (shared ? '&shared=true' : ''),
      { method: 'POST', body: file },
    ),
  /** The asset library, spanning the workspace's own, the shared `common`
   * one and any example library - each entry tagged with which. */
  listAssets: () =>
    request<{
      workspace: string
      assets: AssetFile[]
      folders: string[]
      libraries: AssetLibrary[]
      shadowed: ShadowedAsset[]
    }>('/api/assets'),
  /** Download a multi-file asset selection as one zip - the gallery's bulk
   * download, for the input side. Spans every library on the search path,
   * since the grid does. */
  archiveAssets: archiveFrom('/api/assets/archive'),
  /** Permanently remove one asset. Answers 403 for one an examples tree
   * brought with it, which is not this server's to delete. */
  deleteAsset: (name: string) =>
    request<{ name: string; deleted: boolean; origin: string }>(
      `/api/assets/${encodePath(name)}`,
      { method: 'DELETE' },
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
        usage?: { files: number; bytes: number }
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
    request<{
      name: string
      workspace: string
      origin: string
      warnings: string[]
    }>(`/api/workflows/${encodePath(name)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workflow }),
    }),
  listPrompts: () =>
    request<{
      /** The search path in order; the writable workspace root is where a
       * save lands. */
      libraries: LibraryRoot[]
      shadowed: ShadowedEntry[]
      prompts: string[]
      details: Record<string, PromptDetail>
    }>('/api/prompts'),
  /** The prompt plus which library it came from, read off the response
   * headers rather than a separate `listPrompts` lookup. */
  getPrompt: (name: string) =>
    fetchJson<PromptDefinition>(`/api/prompts/${encodePath(name)}`).then(
      ({ body, response }): StoredPrompt => ({
        prompt: body,
        origin: response.headers.get('X-Prompt-Origin') ?? '',
        writable: response.headers.get('X-Prompt-Writable') !== 'false',
      }),
    ),
  savePrompt: (name: string, prompt: PromptDefinition) =>
    request<{ name: string }>(`/api/prompts/${encodePath(name)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    }),
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

export const TERMINAL_STATUSES = ['succeeded', 'failed', 'cancelled']

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
