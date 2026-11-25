import type {
  AssetDeleted,
  NoteRead,
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
  PromptList,
  RestartBatchRerun,
  ServerInfo,
  TaskList,
  BuiltinWorkflows,
  Uploaded,
  ValidationResult,
  WorkflowDefinition,
  WorkflowDeleted,
  WorkflowList,
  WorkspaceDeleted,
  WorkspaceInfo,
  WorkspaceChanges,
  WorkspaceList,
} from './types'
import { ApiError, errorDetail } from './apiErrors'
import {
  archiveFrom,
  fetchText,
  isValidateRequest,
  jsonInit,
  preconditionHeaders,
  request,
  type JobSubmitBody,
  type RequestWriteBody,
  type ValidateRequest,
  type WritePreconditions,
} from './apiRequest'
import {
  addToken,
  appendQuery,
  encodePath,
  linked,
  outputUrl,
  queryOf,
  scopeTo,
  segment,
  withToken,
} from './apiUrls'
import { documentRoutes } from './apiDocuments'
import { adoptJob } from './jobWatch'

/** A job the server accepted is watched from now, so one that ends before
 * the next status poll still raises its finish toast (jobWatch.ts). */
function adopting(job: JobDetail): JobDetail {
  if (job.id) adoptJob(job.id)
  return job
}

// Pages and their test mocks import these from here
export { ApiError, errorDetail, outputUrl }

export type {
  JobSubmitBody,
  RequestWriteBody,
  ValidateRequest,
  WritePreconditions,
}

// The ledger's routes (request notes, the change counters): workspace in the path
const UNSCOPED = { scope: false }
const ws = encodeURIComponent
const notesUrl = (w: string) => `/api/workspaces/${ws(w)}/request-notes`
const ledgerRead = <T>(url: string) => request<T>(url, undefined, UNSCOPED)

export const api = {
  // getWorkflow, saveWorkflow, getPrompt, savePrompt
  ...documentRoutes,
  // The three library listings ask for each entry's times (`times=true`)
  listWorkflows: (o?: { inputs?: boolean }) =>
    request<WorkflowList>(`/api/workflows?inputs=${!!o?.inputs}&times=true`),
  // Unscoped on purpose: `workspace` is explicit so Status can span every
  // workspace and a workspace's Jobs page can name its own. `status` is one
  // state or a set ('succeeded,failed,cancelled'); `limit` keeps the newest N.
  // `requestId` (with `item`) narrows to the jobs linked to a requests-ledger
  // request, or one item of it; `opts.prompts` adds each job's prompt
  listJobs: (
    workspace?: string,
    limit?: number,
    status?: string,
    requestId?: string,
    item?: string,
    opts?: { prompts?: boolean },
  ) => {
    const link = linked(requestId, item)
    const qs = queryOf({ workspace, status, ...link, limit, ...opts })
    const url = qs ? `/api/jobs?${qs}` : '/api/jobs'
    return request<JobList>(url, undefined, UNSCOPED)
  },
  getJob: (id: string) => request<JobDetail>(`/api/jobs/${segment(id)}`),
  /** The definition a job ran, for the job page's read-only flow view.
   * `realized` true means `definition` is the realized copy the run itself
   * wrote - every mutable input pinned - rather than the definition as
   * submitted. 404s when the job named a workflow file that is no longer
   * readable. */
  getJobWorkflow: (id: string) =>
    request<JobWorkflow>(`/api/jobs/${segment(id)}/workflow`),
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
    const path = `/api/jobs/${segment(id)}/event-log`
    return request<EventLogReply>(qs ? `${path}?${qs}` : path)
  },
  /** Queue the job again. `newSeed` draws a fresh seed into the workflow's
   * seed variable; without it the arguments repeat exactly, which the step
   * cache serves from the earlier run rather than generating anything.
   * `arguments` override the run's own, and are sent only when there are
   * some - the server stores them as the rerun's `overrides` exactly as
   * sent, so they must be only what changed. */
  rerunJob: async (
    id: string,
    {
      newSeed = false,
      arguments: overrides,
    }: { newSeed?: boolean; arguments?: Record<string, unknown> } = {},
  ) =>
    adopting(
      await request<JobDetail>(`/api/jobs/${segment(id)}/rerun`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(
          overrides && Object.keys(overrides).length
            ? { new_seed: newSeed, arguments: overrides }
            : { new_seed: newSeed },
        ),
      }),
    ),
  /** Rerun every job one dead server's restart cancelled, naming any one of
   * them. Without `acknowledged` it is a priced preview - `would_queue` and
   * `skipped` - and queues nothing; with it the batch is queued. Every
   * member is priced, so only a click calls this, never a poll. Unscoped:
   * a batch can span workspaces and the route needs no workspace. */
  rerunRestartBatch: (id: string, acknowledged = false) =>
    request<RestartBatchRerun>(
      `/api/jobs/${segment(id)}/rerun-restart-batch`,
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
    request<JobMoved>(`/api/jobs/${segment(id)}/move`, {
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
        scopeTo(`/api/jobs/${segment(id)}/export`, workspace),
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
    request<JobCancelled>(`/api/jobs/${segment(id)}/cancel`, {
      method: 'POST',
    }),
  /** Queue a run. With `request` and `item` it is linked to that item of
   * the requests ledger (422 `request_closed`, `item_dropped`,
   * `unknown_request`, `unknown_item` before anything queues); with
   * `acknowledged_cost` bound to a validate plan, a run whose plan has
   * moved is a 409 carrying the current plan and what to resend. */
  submitJob: async (body: JobSubmitBody) =>
    adopting(
      await request<JobDetail>('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }),
    ),
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
    request<ModelDownload>(`/api/models/downloads/${segment(id)}/cancel`, {
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
  // the limit just needs to exceed any real output directory's file count.
  // `roles` narrows server-side (repeated `role` parameters; a null role
  // counts as final there); the page filters on the client, so asks for none
  // `requestId` narrows to the runs whose job is linked to that
  // requests-ledger request (in the selected workspace), and `item` to one
  // item of it. `opts`: `prompts` (by run id), `media` (durations)
  gallery: (
    roles?: string[],
    requestId?: string,
    item?: string,
    opts?: { prompts?: boolean; media?: boolean },
  ) => {
    const link = linked(requestId, item)
    const qs = queryOf({ limit: 100000, role: roles, ...link, ...opts })
    return request<GalleryList>(`/api/gallery?${qs}`)
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
  /** A gallery text file's contents, from the `url` its listing gave -
   * which names its workspace already */
  fileText: (url: string) => fetchText(url, { scope: false }),
  /** Run every assessment probe that applies to one output, in the
   * server's threadpool: a GET, scoped to the picker's workspace. It decodes
   * the whole file, so only a click calls it, never a poll. */
  assess: (name: string) =>
    request<AssessReply>(`/api/gallery/${encodePath(name)}/assess`),
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
   * library every workspace under this root shares. `createOnly` refuses
   * (409 `exists`) to replace an asset already under that name. */
  uploadMedia: (
    file: File,
    assetName?: string,
    shared = false,
    pre?: WritePreconditions,
  ) =>
    request<Uploaded>(
      `/api/uploads?filename=${encodeURIComponent(file.name)}` +
        (assetName ? `&asset_name=${encodeURIComponent(assetName)}` : '') +
        (shared ? '&shared=true' : ''),
      { method: 'POST', body: file, headers: preconditionHeaders(pre) },
    ),
  /** The asset library: the workspace's own, the shared `common` one and any
   * example library, each entry tagged with which; `media` adds A/V lengths */
  listAssets: (opts?: { media?: boolean; limit?: number }) =>
    request<AssetList>(opts ? `/api/assets?${queryOf(opts)}` : '/api/assets'),
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
      `/api/workspaces/${encodeURIComponent(workspace)}/requests?needs=true`,
      undefined,
      { scope: false },
    ),
  getRequest: (workspace: string, id: string) =>
    request<LedgerRequest>(
      `/api/workspaces/${encodeURIComponent(workspace)}/requests/${encodeURIComponent(id)}?reasons=true`,
      undefined,
      { scope: false },
    ),
  /** Write the whole plan back as read, at the revision it was read at,
   * by the user. A stale revision is a 409 ApiError. */
  putRequest: (workspace: string, id: string, body: RequestWriteBody) =>
    request<LedgerRequest>(
      `/api/workspaces/${encodeURIComponent(workspace)}/requests/${encodeURIComponent(id)}?reasons=true`,
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
  notesChannel: (w: string) =>
    ledgerRead<NotesChannelState>(`/api/workspaces/${ws(w)}/notes-channel`),
  /** What moved: the change counters the change feed reads (changes.svelte.ts) */
  workspaceChanges: (w: string) =>
    ledgerRead<WorkspaceChanges>(`/api/workspaces/${ws(w)}/changes`),
  listWorkspaces: () => request<WorkspaceList>('/api/workspaces?times=true'),
  /** The target's own note and the etag `setNote`'s `ifMatch` names: the
   * precondition a save reads, not a view (the page's resource is that).
   * Scoped like `setNote`. */
  noteForSave: (target: string, workspace?: string) =>
    noteRequest<NoteRead>(
      appendQuery('/api/notes', 'target', target),
      workspace,
    ),
  /** Whether the server lets the saved token in: a read any token opens */
  checkToken: async (): Promise<void> => {
    await request<unknown>('/api/workspaces', undefined, UNSCOPED)
  },
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
    request<PipelineDescription>(`/api/pipelines/${segment(name)}`),
  listClasses: (kind: string) =>
    request<ClassList>(`/api/classes?kind=${encodeURIComponent(kind)}`),
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
  listPrompts: () => request<PromptList>('/api/prompts?times=true'),
  /** Catalog LoRA entries for one base: `model` a Hub repo id (or a saved
   * workflow's name), `workflow` an optional partition. Catalog-only - the
   * Hub is searched by /api/loras/recommend, which the UI never calls.
   * Rejected entries are included; the caller drops them. */
  listLoras: ({ model, workflow }: { model: string; workflow?: string }) => {
    const query = new URLSearchParams({ model })
    if (workflow) query.set('workflow', workflow)
    return request<LoraList>(`/api/loras?${query}`)
  },
  deletePrompt: (name: string) =>
    request<Deleted>(`/api/prompts/${encodePath(name)}`, { method: 'DELETE' }),
  promptDownloadUrl: (name: string) =>
    withToken(`/api/prompts/${encodePath(name)}/download`),
  getPromptSchema: () => request<Record<string, unknown>>('/api/prompt-schema'),
  listEnhancers: () => request<EnhancerPresets>('/api/enhancers'),
  enhance: async (body: {
    idea: string
    preset: string
    model_name?: string
    device?: string
  }) =>
    adopting(
      await request<JobDetail>('/api/enhance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }),
    ),
}

export { fetchOutputText } from './apiRequest'

/** Set (or, with null, clear) the note on a workspace, asset, run or file.
 * A file name or run directory is read in the picked workspace, unless
 * `workspace` names another - a job page edits the run of the job's own
 * workspace, wherever the picker has since moved. */
export function setNote(
  target: string,
  note: string | null,
  workspace?: string,
  ifMatch?: string,
): Promise<NoteSet> {
  const init = jsonInit(
    'PUT',
    { target, note },
    preconditionHeaders({ ifMatch }),
  )
  return noteRequest<NoteSet>('/api/notes', workspace, init)
}

/** A notes route, in `workspace` or else the picker's */
const noteRequest = <T>(path: string, workspace?: string, init?: RequestInit) =>
  request<T>(workspace === undefined ? path : scopeTo(path, workspace), init, {
    scope: workspace === undefined,
  })

export const TERMINAL_STATUSES = ['succeeded', 'failed', 'cancelled']

/** Stream a job's events; returns a stop function. The stream closes itself
 * when a terminal job_status arrives; transient errors are left alone so
 * EventSource reconnects and resumes losslessly via Last-Event-ID. */
export function streamJobEvents(
  jobId: string,
  after: number,
  onEvent: (event: JobEvent) => void,
  onEnd: () => void,
  onDrop?: () => void,
): () => void {
  // EventSource cannot set custom headers, so a configured token rides
  // along as a query parameter for this one route - see docs/SERVER.md.
  const source = new EventSource(
    withToken(`/api/jobs/${segment(jobId)}/events?after=${after}`),
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
  // A dropped connection is EventSource's own to repair, so it is not closed
  // here - closing froze live progress on any transient hiccup. `onDrop`
  // lets the caller read the job again: after a restart it may have ended,
  // or be gone, and no event will ever say so.
  source.onerror = () => onDrop?.()
  return () => source.close()
}
