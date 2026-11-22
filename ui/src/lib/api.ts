import type {
  AssetDeleted,
  NoteSet,
  StarSet,
  AssetList,
  ClassList,
  Deleted,
  DiffusersStatus,
  DiskCacheCleared,
  DiskCacheEntryDeleted,
  DiskCacheList,
  EnhancerPresets,
  EventLogReply,
  GalleryList,
  GalleryMetadata,
  AssessReply,
  HealthInfo,
  JobCancelled,
  JobDetail,
  JobEvent,
  JobExport,
  JobList,
  JobMoved,
  JobWorkflow,
  Kept,
  LedgerDeleted,
  LedgerList,
  LedgerRequest,
  LoraList,
  NoteAdd,
  NoteDeleted,
  NoteEdit,
  NotesChannelState,
  NoteWritten,
  MemoryInfo,
  ModelCache,
  ModelDeleted,
  ModelDownload,
  ModelDownloads,
  OutputDeleted,
  PipelineDescription,
  PipelineNames,
  PromptDefinition,
  PromptList,
  PromptSaved,
  RestartBatchRerun,
  ServerInfo,
  StoredPrompt,
  TaskList,
  BuiltinWorkflows,
  Uploaded,
  ValidationResult,
  WorkflowDefinition,
  WorkflowDeleted,
  WorkflowList,
  WorkflowSaved,
  WorkflowWithOrigin,
  WorkspaceDeleted,
  WorkspaceInfo,
  WorkspaceChanges,
  WorkspaceList,
} from './types'
import { ApiError, errorDetail } from './apiErrors'
import {
  archiveFrom,
  fetchJson,
  isValidateRequest,
  jsonInit,
  request,
  type JobSubmitBody,
  type RequestWriteBody,
  type ValidateRequest,
} from './apiRequest'
import {
  addToken,
  appendQuery,
  encodePath,
  outputUrl,
  scopeTo,
  withToken,
  galleryThumbnailUrl,
} from './apiUrls'

// Pages and their test mocks import these from here
export { ApiError, errorDetail, outputUrl }

export type { JobSubmitBody, RequestWriteBody, ValidateRequest }

// The ledger's routes (request notes, the change counters): workspace in the path
const UNSCOPED = { scope: false }
const wsUrl = (workspace: string, rest: string) =>
  `/api/workspaces/${encodeURIComponent(workspace)}/${rest}`
const notesUrl = (workspace: string) => wsUrl(workspace, 'request-notes')
const ledgerRead = <T>(workspace: string, rest: string) =>
  request<T>(wsUrl(workspace, rest), undefined, UNSCOPED)

export const api = {
  listWorkflows: () => request<WorkflowList>('/api/workflows'),
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
  // `requestId` (with `item`) narrows to the jobs linked to a requests-ledger
  // request, or one item of it
  listJobs: (
    workspace?: string,
    limit?: number,
    status?: string,
    requestId?: string,
    item?: string,
  ) => {
    const query = new URLSearchParams()
    if (workspace) query.set('workspace', workspace)
    if (status) query.set('status', status)
    if (requestId) query.set('request', requestId)
    if (requestId && item) query.set('item', item)
    if (limit) query.set('limit', String(limit))
    const qs = query.toString()
    return request<JobList>(qs ? `/api/jobs?${qs}` : '/api/jobs', undefined, {
      scope: false,
    })
  },
  getJob: (id: string) => request<JobDetail>(`/api/jobs/${id}`),
  /** The definition a job ran, for the job page's read-only flow view.
   * `realized` true means `definition` is the realized copy the run itself
   * wrote - every mutable input pinned - rather than the definition as
   * submitted. 404s when the job named a workflow file that is no longer
   * readable. */
  getJobWorkflow: (id: string) =>
    request<JobWorkflow>(`/api/jobs/${id}/workflow`),
  /** One JSON page of a job's events, after `after` (exclusive). `kinds`
   * goes as repeated parameters and keeps events whose `event` or `kind`
   * is one of them. A finished job answers from the tail history kept,
   * with a `note` on what was dropped; an unknown job is a 404. */
  eventLog: (
    id: string,
    {
      after,
      limit,
      kinds,
    }: { after?: number; limit?: number; kinds?: string[] } = {},
  ) => {
    const query = new URLSearchParams()
    if (after !== undefined) query.set('after', String(after))
    if (limit !== undefined) query.set('limit', String(limit))
    for (const kind of kinds ?? []) query.append('kinds', kind)
    const qs = query.toString()
    const path = `/api/jobs/${id}/event-log`
    return request<EventLogReply>(qs ? `${path}?${qs}` : path)
  },
  /** Queue the job again. `newSeed` draws a fresh seed into the workflow's
   * seed variable; without it the arguments repeat exactly, which the step
   * cache serves from the earlier run rather than generating anything.
   * `arguments` override the run's own, and are sent only when there are
   * some - the server stores them as the rerun's `overrides` exactly as
   * sent, so they must be only what changed. */
  rerunJob: (
    id: string,
    {
      newSeed = false,
      arguments: overrides,
    }: { newSeed?: boolean; arguments?: Record<string, unknown> } = {},
  ) =>
    request<JobDetail>(`/api/jobs/${id}/rerun`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(
        overrides && Object.keys(overrides).length
          ? { new_seed: newSeed, arguments: overrides }
          : { new_seed: newSeed },
      ),
    }),
  /** Rerun every job one dead server's restart cancelled, naming any one of
   * them. Without `acknowledged` it is a priced preview - `would_queue` and
   * `skipped` - and queues nothing; with it the batch is queued. Every
   * member is priced, so only a click calls this, never a poll. Unscoped:
   * a batch can span workspaces and the route needs no workspace. */
  rerunRestartBatch: (id: string, acknowledged = false) =>
    request<RestartBatchRerun>(
      `/api/jobs/${id}/rerun-restart-batch`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(acknowledged ? { acknowledged_cost: true } : {}),
      },
      { scope: false },
    ),
  listTasks: () => request<TaskList>('/api/tasks'),
  listBuiltinWorkflows: () =>
    request<BuiltinWorkflows>('/api/builtin-workflows'),
  describeTask: (command: string) =>
    request<PipelineDescription>(`/api/tasks/${encodeURIComponent(command)}`),
  moveJob: (id: string, direction: 'up' | 'down' | 'front' | 'back') =>
    request<JobMoved>(`/api/jobs/${id}/move`, {
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
    request<JobExport>(
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
    request<JobCancelled>(`/api/jobs/${id}/cancel`, {
      method: 'POST',
    }),
  /** Queue a run. With `request` and `item` it is linked to that item of
   * the requests ledger (422 `request_closed`, `item_dropped`,
   * `unknown_request`, `unknown_item` before anything queues); with
   * `acknowledged_cost` bound to a validate plan, a run whose plan has
   * moved is a 409 carrying the current plan and what to resend. */
  submitJob: (body: JobSubmitBody) =>
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
  listDownloads: () => request<ModelDownloads>('/api/models/downloads'),
  cancelDownload: (id: string) =>
    request<ModelDownload>(`/api/models/downloads/${id}/cancel`, {
      method: 'POST',
    }),
  deleteModel: (repo: string) =>
    request<ModelDeleted>(`/api/models?repo=${encodeURIComponent(repo)}`, {
      method: 'DELETE',
    }),
  diffusersStatus: () => request<DiffusersStatus>('/api/system/diffusers'),
  updateDiffusers: () =>
    request<DiffusersStatus>('/api/system/diffusers/update', {
      method: 'POST',
    }),
  health: () => request<HealthInfo>('/api/health'),
  server: () => request<ServerInfo>('/api/server'),
  /** dw's own disk caches (dw/disk_caches.py) - Inductor's and Triton's
   * compiled kernels - with each one's size and whether it can be cleared. */
  listCaches: () => request<DiskCacheList>('/api/caches'),
  /** Empty one cache's root. 409 (an `ApiError`) while a job is running or
   * queued; 400 when the root is not dw's to clear. */
  clearCache: (name: string) =>
    request<DiskCacheCleared>(`/api/caches/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),
  /** Delete one entry of a cache that has them (today `quantized`). 404 for
   * an unknown entry; 409 (an `ApiError`) while a job or download is active. */
  deleteCacheEntry: (name: string, entry: string) =>
    request<DiskCacheEntryDeleted>(
      `/api/caches/${encodeURIComponent(name)}/entries/${encodeURIComponent(entry)}`,
      { method: 'DELETE' },
    ),
  // Loads the whole gallery in one request, like listWorkflows/listPrompts -
  // the limit just needs to exceed any real output directory's file count
  // `roles` narrows to those roles server-side (repeated `role` parameters;
  // a null role counts as final there). The page filters on the client, so
  // it asks for none
  // `requestId` narrows to the runs whose job is linked to that
  // requests-ledger request (in the selected workspace), and `item` to one
  // item of it
  gallery: (roles?: string[], requestId?: string, item?: string) => {
    const params = new URLSearchParams({ limit: '100000' })
    for (const role of roles ?? []) params.append('role', role)
    if (requestId) params.set('request', requestId)
    if (requestId && item) params.set('item', item)
    return request<GalleryList>(`/api/gallery?${params}`)
  },
  // `workspace`, when given, names the file's own workspace and wins over
  // whatever is currently selected in the picker - mirrors `outputUrl`, since
  // a job page must read its own files from where they were written
  galleryMetadata: (name: string, workspace?: string) =>
    request<GalleryMetadata>(
      workspace === undefined
        ? `/api/gallery/${encodePath(name)}/metadata`
        : scopeTo(`/api/gallery/${encodePath(name)}/metadata`, workspace),
      undefined,
      { scope: workspace === undefined },
    ),
  /** Run every assessment probe that applies to one output, in the
   * server's threadpool: a GET, scoped to the picker's workspace. It decodes
   * the whole file, so only a click calls it, never a poll. */
  assess: (name: string) =>
    request<AssessReply>(`/api/gallery/${encodePath(name)}/assess`),
  galleryThumbnailUrl,
  deleteOutput: (name: string) =>
    request<OutputDeleted>(`/api/gallery/${encodePath(name)}`, {
      method: 'DELETE',
    }),
  setStarred: (name: string, starred: boolean) =>
    request<StarSet>('/api/stars', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, starred }),
    }),
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
    request<Uploaded>(
      `/api/uploads?filename=${encodeURIComponent(file.name)}` +
        (assetName ? `&asset_name=${encodeURIComponent(assetName)}` : '') +
        (shared ? '&shared=true' : ''),
      { method: 'POST', body: file },
    ),
  /** The asset library, spanning the workspace's own, the shared `common`
   * one and any example library - each entry tagged with which. */
  listAssets: () => request<AssetList>('/api/assets'),
  /** Download a multi-file asset selection as one zip - the gallery's bulk
   * download, for the input side. Spans every library on the search path,
   * since the grid does. */
  archiveAssets: archiveFrom('/api/assets/archive'),
  /** Permanently remove one asset. Answers 403 for one an examples tree
   * brought with it, which is not this server's to delete. */
  deleteAsset: (name: string) =>
    request<AssetDeleted>(`/api/assets/${encodePath(name)}`, {
      method: 'DELETE',
    }),
  // The requests ledger. Unscoped: the workspace is in the path
  listRequests: (workspace: string) =>
    request<LedgerList>(
      `/api/workspaces/${encodeURIComponent(workspace)}/requests`,
      undefined,
      { scope: false },
    ),
  getRequest: (workspace: string, id: string) =>
    request<LedgerRequest>(
      `/api/workspaces/${encodeURIComponent(workspace)}/requests/${encodeURIComponent(id)}`,
      undefined,
      { scope: false },
    ),
  /** Write the whole plan back as read, at the revision it was read at,
   * by the user. A stale revision is a 409 ApiError. */
  putRequest: (workspace: string, id: string, body: RequestWriteBody) =>
    request<LedgerRequest>(
      `/api/workspaces/${encodeURIComponent(workspace)}/requests/${encodeURIComponent(id)}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ by: 'user', ...body }),
      },
      { scope: false },
    ),
  /** Remove a request from the ledger, at the revision it was read at. Its
   * jobs and their files stay. 409 `stale_revision`, or
   * `request_has_active_jobs` while one of its jobs is queued or running. */
  deleteRequest: (workspace: string, id: string, revision: number) =>
    request<LedgerDeleted>(
      `/api/workspaces/${encodeURIComponent(workspace)}/requests/${encodeURIComponent(id)}?revision=${revision}`,
      { method: 'DELETE' },
      { scope: false },
    ),
  /** Add one of the user's notes for the agent, on an item or (no `item`)
   * on the request. 422 `invalid_note_ref`, `too_many_refs`,
   * `too_many_notes`, `unknown_item`, `item_dropped`. */
  addNote: (workspace: string, body: NoteAdd) =>
    request<NoteWritten>(notesUrl(workspace), jsonInit('POST', body), UNSCOPED),
  /** A note's new text; its references stay. 404 `unknown_note`. */
  editNote: (workspace: string, body: NoteEdit) =>
    request<NoteWritten>(notesUrl(workspace), jsonInit('PUT', body), UNSCOPED),
  deleteNote: (workspace: string, requestId: string, note: string) =>
    request<NoteDeleted>(
      `${notesUrl(workspace)}?${new URLSearchParams({ request: requestId, note })}`,
      { method: 'DELETE' },
      UNSCOPED,
    ),
  /** Whether an agent session's notes channel polled the workspace lately */
  notesChannel: (workspace: string) =>
    ledgerRead<NotesChannelState>(workspace, 'notes-channel'),
  /** What moved: the change counters the change feed reads (changes.svelte.ts) */
  workspaceChanges: (workspace: string) =>
    ledgerRead<WorkspaceChanges>(workspace, 'changes'),
  listWorkspaces: () => request<WorkspaceList>('/api/workspaces'),
  createWorkspace: (name: string) =>
    request<WorkspaceInfo>('/api/workspaces', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }),
  /** Delete a workspace and everything in it. The server refuses without
   * `acknowledged`, answering with what it would remove - so the caller can
   * show that before asking again. */
  deleteWorkspace: (name: string, acknowledged = false) =>
    request<WorkspaceDeleted>(
      `/api/workspaces/${encodeURIComponent(name)}?acknowledged=${acknowledged}`,
      { method: 'DELETE' },
    ),
  /** Keep a generated file as an input asset under a stable name. The copy
   * happens on the server, inside the workspace - nothing is downloaded and
   * re-uploaded to reuse a render. */
  keepOutput: (name: string, assetName?: string, overwrite = false) =>
    request<Kept>('/api/assets/keep', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        asset_name: assetName ?? null,
        overwrite,
      }),
    }),
  listPipelines: () => request<PipelineNames>('/api/pipelines'),
  describePipeline: (name: string) =>
    request<PipelineDescription>(`/api/pipelines/${name}`),
  listClasses: (kind: string) =>
    request<ClassList>(`/api/classes?kind=${kind}`),
  describeClass: (name: string, target: 'call' | 'init' | 'load') =>
    request<PipelineDescription>(
      `/api/classes/${encodeURIComponent(name)}?target=${target}`,
    ),
  getSchema: () => request<Record<string, unknown>>('/api/schema'),
  /** Check a workflow without queuing it. The editor hands over the inline
   * definition it holds; the run page asks by name with the arguments a
   * Run would send, and `sizes: false` to skip the hub for a faster answer.
   * `signal` lets a caller drop a check it no longer needs. */
  validate: (
    input: WorkflowDefinition | ValidateRequest,
    signal?: AbortSignal,
  ) => {
    const asked: ValidateRequest = isValidateRequest(input)
      ? input
      : { workflow: input }
    const { sizes, ...body } = asked
    // one literal, so tests/test_ui_twins.py reads the route with its POST
    const route = '/api/validate'
    const path = sizes === false ? appendQuery(route, 'sizes', 'false') : route
    return request<ValidationResult>(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    })
  },
  deleteWorkflow: (name: string) =>
    request<WorkflowDeleted>(`/api/workflows/${encodePath(name)}`, {
      method: 'DELETE',
    }),
  workflowDownloadUrl: (name: string) =>
    withToken(`/api/workflows/${encodePath(name)}/download`),
  saveWorkflow: (name: string, workflow: WorkflowDefinition) =>
    request<WorkflowSaved>(`/api/workflows/${encodePath(name)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workflow }),
    }),
  listPrompts: () => request<PromptList>('/api/prompts'),
  /** Catalog LoRA entries for one base: `model` a Hub repo id (or a saved
   * workflow's name), `workflow` an optional partition. Catalog-only - the
   * Hub is searched by /api/loras/recommend, which the UI never calls.
   * Rejected entries are included; the caller drops them. */
  listLoras: ({ model, workflow }: { model: string; workflow?: string }) => {
    const query = new URLSearchParams({ model })
    if (workflow) query.set('workflow', workflow)
    return request<LoraList>(`/api/loras?${query}`)
  },
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
    request<PromptSaved>(`/api/prompts/${encodePath(name)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    }),
  deletePrompt: (name: string) =>
    request<Deleted>(`/api/prompts/${encodePath(name)}`, { method: 'DELETE' }),
  promptDownloadUrl: (name: string) =>
    withToken(`/api/prompts/${encodePath(name)}/download`),
  getPromptSchema: () => request<Record<string, unknown>>('/api/prompt-schema'),
  listEnhancers: () => request<EnhancerPresets>('/api/enhancers'),
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

/** Set (or, with null, clear) the note on a workspace, asset, run or file.
 * A file name or run directory is read in the picked workspace, unless
 * `workspace` names another - a job page edits the run of the job's own
 * workspace, wherever the picker has since moved. */
export function setNote(
  target: string,
  note: string | null,
  workspace?: string,
): Promise<NoteSet> {
  const path = '/api/notes'
  return request<NoteSet>(
    workspace === undefined ? path : scopeTo(path, workspace),
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target, note }),
    },
    { scope: workspace === undefined },
  )
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
