<script lang="ts">
  import { proofByRun } from '../proofs'
  import JobRow from '../job/JobRow.svelte'
  import { resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import { onMount } from 'svelte'
  import { Inbox } from '@lucide/svelte'
  import { api, TERMINAL_STATUSES } from '../api'
  import { action } from '../action.svelte'
  import { codec, hashState } from '../router.svelte'
  import RequestFilter from '../request/RequestFilter.svelte'
  import {
    JOBS_PAGE,
    heldFor,
    type HeldJobs,
    jobMatches,
    listWithPrompts,
    listingRefused,
    listingWorkspace,
    queueFirst,
    queueLength,
    queuedEverywhere,
    runningProgress,
  } from '../jobsListing'
  import { loadWorkspaces, workspace } from '../workspace.svelte'
  import { wsHref } from '../routes'
  import {
    rerunRestartBatchWithConfirm,
    restartBanners,
    restartReader,
    type RestartBanner,
  } from '../restartBatches'

  let { scope = 'workspace' }: { scope?: 'workspace' | 'all' } = $props()

  // The filters live in the hash's query, so opening a job and coming back
  // - or sharing the link - keeps the narrowed view
  // A status the picker does not offer (a stale or hand-edited link) reads as
  // all: it would otherwise hide every row under a select showing nothing
  const STATUSES = ['running', 'queued', ...TERMINAL_STATUSES]
  const statusFilter = hashState(
    'status',
    '',
    codec.oneOf<string>(['', ...STATUSES]),
  )
  const nameFilter = hashState('q', '')
  // Server-side, unlike the two filters above: it changes which jobs are
  // fetched at all, not just which of the fetched ones are shown. Only read
  // under scope 'all' - empty there means every workspace, which is the
  // point of that view; a workspace's own Jobs page filters by
  // `workspace.current` instead.
  const workspaceFilter = hashState('ws', '')
  // Server-side too: the request (or one item of it) the jobs were linked
  // to. RequestFilter keeps it in the hash (`req`) and reads it at init
  const requestScope = $state({ request: '', item: '' })

  // The server-side filters narrow what is fetched, so an empty answer is
  // "nothing matches"; the client-side ones only mean that when there were
  // rows to filter (on an empty listing the answer is "no jobs yet")
  const narrowed = $derived(
    !!requestScope.request || (scope === 'all' && !!workspaceFilter.value),
  )
  function clearFilters() {
    statusFilter.value = ''
    nameFilter.value = ''
    workspaceFilter.value = ''
    requestScope.request = ''
    requestScope.item = ''
  }
  // A request lives in one workspace, so its filter is never read across
  // them (listingWorkspace): on the all-jobs view it pins the picker to the
  // workspace it is read in, and choosing every workspace lets go of it
  const listed = $derived(
    listingWorkspace(
      scope,
      workspaceFilter.value,
      workspace.current,
      requestScope.request,
    ),
  )
  $effect(() => {
    if (scope === 'all' && listed !== workspaceFilter.value)
      workspaceFilter.value = listed
  })

  // Only worth a picker once there is a choice - a single-workspace server
  // has nothing to filter by
  onMount(loadWorkspaces)

  // Live state, as it was: the listing the filters name, read every 3 s.
  // What load reads before its first await (the scope's workspace and the
  // request filter) restarts the read when it changes, so a narrowed view
  // never shows another question's rows
  // Prompts ride along only for jobs not yet read (listWithPrompts): a
  // prompt never changes, so the 3 s poll does not ask for them again
  const known = new Map<string, string>()
  const held: HeldJobs = { question: '', jobs: new Map() }
  // How far back the finished jobs are read: the newest page, then what Show
  // older or Search all asked for (`all`: the whole history was read once,
  // and from then on a poll reads a page and merges it into `held`, so the
  // cost per poll stays a page). A filter does not reset it; a different
  // workspace or request is a new question and starts at the first page, as
  // `held` does. Derived and overridden by the buttons, so `load` only reads
  const question = $derived(JSON.stringify([listed, requestScope]))
  let paging = $derived.by(() => {
    void question
    return { shown: JOBS_PAGE, all: false }
  })
  // The banners' batch sizes, asked once each rather than every poll
  const restarts = restartReader()
  const list = resource(
    async () => {
      const [ws, scoped] = [listed, { ...requestScope }]
      const { shown: limit, all } = paging
      const { list: answer, prompts } = await listWithPrompts(
        ws,
        scoped,
        known,
        limit,
        all ? heldFor(held, ws, scoped) : undefined,
      )
      const rows = answer.jobs.reverse()
      const queued = await queuedEverywhere(rows, !!(ws || scoped.request))
      const progress = await runningProgress(rows)
      const partly = !all && limit < answer.finished
      const restart = await restarts.read(rows, partly ? { ws, scoped } : null)
      return {
        restart,
        rows,
        total: answer.total,
        finished: answer.finished,
        queueSize: queueLength(rows, queued),
        prompts,
        progress,
      }
    },
    { live: { everyMs: 3000 } },
  )
  // Each run's picture, from the workspace's gallery
  const gallery = resource(
    async () => {
      const { files, workspace: from } = await api.gallery()
      return { from, runs: proofByRun(files) }
    },
    { watch: ['gallery'] },
  )
  const runs = $derived(gallery.data?.runs ?? {})
  const listedGallery = $derived(gallery.data?.from ?? '')
  const prompts = $derived(list.data?.prompts ?? {})
  const progress = $derived(list.data?.progress ?? {})
  // The server turning the question down: its message is the explanation,
  // and the rows of an earlier question must not stay under this one's
  const refused = $derived(
    list.error !== undefined &&
      (list.status === 'error' || list.stale) &&
      listingRefused(list.error),
  )
  const jobs = $derived(refused ? [] : (list.data?.rows ?? []))
  const filtered = $derived(
    narrowed ||
      ((!!statusFilter.value || !!nameFilter.value) && jobs.length > 0),
  )
  const queueSize = $derived(list.data?.queueSize ?? 0)
  const total = $derived(refused ? 0 : (list.data?.total ?? 0))
  const finished = $derived(refused ? 0 : (list.data?.finished ?? 0))
  const partial = $derived(!paging.all && paging.shown < finished)
  // Queued and running jobs are always read whole: a filter on only those
  // searches nothing the page left out
  const liveOnly = $derived(
    statusFilter.value === 'queued' || statusFilter.value === 'running',
  )
  const olderBy = $derived(Math.min(JOBS_PAGE, finished - paging.shown))
  const visible = $derived(
    queueFirst(
      jobs.filter(
        (job) =>
          (!statusFilter.value || job.status === statusFilter.value) &&
          jobMatches(job, prompts[job.id] ?? '', nameFilter.value),
      ),
    ),
  )
  // Batches a preview found nothing left to rerun in: their banner stays
  // hidden for the rest of the session, in memory only - the count may be
  // behind and the preview corrected it
  let dismissedBatches = $state(new Set<string>())
  // Not the name-filtered rows: a batch spans
  // workflows. A status filter that hides cancelled rows hides it too
  const banners = $derived(
    statusFilter.value && statusFilter.value !== 'cancelled'
      ? []
      : restartBanners(
          refused ? [] : (list.data?.restart.tombstones ?? []),
          dismissedBatches,
          list.data?.restart.pending,
        ),
  )

  // Only a click previews: the preview prices every member of the batch,
  // which the 3 s poll must never pay for. Its question and its failures
  // are rerunRestartBatchWithConfirm's
  // Which banner's rerun is in flight: the action's one flag is every
  // banner's, but only that banner is busy
  let busyBatch = $state<string | null>(null)
  const rerunning = action(
    async (banner: RestartBanner) => {
      busyBatch = banner.batch
      const outcome = await rerunRestartBatchWithConfirm(banner.id)
      if (outcome === 'empty')
        dismissedBatches = new Set([...dismissedBatches, banner.batch])
      if (outcome === 'queued') restarts.forget(banner.batch)
      return outcome
    },
    { doing: 'rerun the jobs' },
  )

  // The job being moved, for its buttons' busy cue
  let movingId = $state<string | null>(null)
  const moving = action(
    async (id: string, direction: 'up' | 'down' | 'front') => {
      movingId = id
      await api.moveJob(id, direction)
      // Same listing the poll asks for - refreshing without its filters
      // would flash every workspace's jobs into a view the user had
      // narrowed to one
      await list.reload()
    },
    { doing: 'move the job' },
  )
  // The buttons sit inside the row's link, so a click must not follow it
  function move(event: MouseEvent, id: string, way: 'up' | 'down' | 'front') {
    event.preventDefault()
    void moving.run(id, way)
  }
</script>

{#snippet filters()}
  <select bind:value={statusFilter.value} title="filter by status">
    <option value="">all statuses</option>
    {#each STATUSES as status (status)}
      <option value={status}>{status}</option>
    {/each}
  </select>
  {#if scope === 'all' && (workspace.names?.length ?? 0) > 1}
    <select
      bind:value={workspaceFilter.value}
      onchange={() => {
        if (!workspaceFilter.value) requestScope.request = ''
      }}
      title="filter by workspace"
    >
      <option value="">all workspaces</option>
      {#each workspace.names ?? [] as name (name)}
        <option value={name}>{name}</option>
      {/each}
    </select>
  {/if}
  <input
    class="filter"
    placeholder="search names, prompts, notes and ids…"
    title="search names, prompts, notes and ids"
    bind:value={nameFilter.value}
  />
{/snippet}
<!-- Under the status page, the queue is a section of the server's page -->
<PageHeader
  title={scope === 'all' ? 'All jobs' : 'Jobs'}
  nested={scope === 'all'}
  filter={filters}
>
  {#snippet actions()}
    <RequestFilter
      bind:request={requestScope.request}
      bind:item={requestScope.item}
      workspace={listed}
    />
  {/snippet}
</PageHeader>

{#each banners as banner (banner.batch)}
  <div class="panel restart" role="status" data-batch={banner.batch}>
    <span>
      {`${banner.count} ${banner.count === 1 ? 'job was' : 'jobs were'} cancelled by a server restart`}
    </span>
    <button
      onclick={() => void rerunning.run(banner)}
      disabled={rerunning.busy && busyBatch === banner.batch}
      aria-busy={rerunning.busy && busyBatch === banner.batch}
      title="preview what a rerun would queue, with its cost, before anything is queued"
    >
      Rerun…
    </button>
  </div>
{/each}

{#if partial && !liveOnly && (statusFilter.value || nameFilter.value)}
  <p class="hint">
    Searching the newest {jobs.length} of {total} jobs.
    <button
      class="quiet"
      onclick={() => {
        paging = { shown: finished, all: true }
      }}
    >
      Search all {total}
    </button>
  </p>
{/if}

<LoadState
  of={list}
  label="jobs"
  empty={!refused && visible.length === 0}
  {filtered}
  onclear={clearFilters}
>
  {#snippet emptyIcon()}<Inbox size={36} strokeWidth={1.5} />{/snippet}
  {#snippet emptyText()}No jobs yet.
    <a href={wsHref(workspace.current, 'workflows')}>Pick a workflow</a> and run it.{/snippet}
  {#if visible.length > 0}
    <div class="panel list">
      {#each visible as job (job.id)}
        {@const proof = job.run_id ? runs[job.run_id] : undefined}
        <JobRow
          {job}
          {queueSize}
          showWorkspace={scope === 'all' && (workspace.names?.length ?? 0) > 1}
          prompt={prompts[job.id] ?? ''}
          percent={progress[job.id] ?? null}
          proof={listedGallery && job.workspace !== listedGallery
            ? undefined
            : (proof ?? null)}
          onmove={move}
          busyId={moving.busy ? movingId : null}
        />
      {/each}
    </div>
  {/if}
</LoadState>

{#if partial}
  <p class="hint">
    Showing the newest {jobs.length} of {total} jobs.
    <button
      class="quiet"
      onclick={() => (paging = { ...paging, shown: paging.shown + JOBS_PAGE })}
    >
      Show {olderBy} older
    </button>
  </p>
{/if}

<style>
  select {
    max-width: 150px;
  }
  .restart {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.55rem 0.8rem;
    margin-bottom: 0.6rem;
    box-shadow: inset 3px 0 0 var(--warn);
  }
</style>
