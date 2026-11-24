<script lang="ts">
      import { proofByRun } from '../proofs'
  import JobRow from '../job/JobRow.svelte'
  import { resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import { onMount } from 'svelte'
  import { Inbox } from '@lucide/svelte'
  import { api } from '../api'
  import { action } from '../action.svelte'
  import { hashState } from '../router.svelte'
  import RequestFilter from '../gallery/RequestFilter.svelte'
  import {
    jobMatches,
    listWithPrompts,
    listingRefused,
    listingWorkspace,
    queueLength,
    queuedEverywhere,
    runningProgress,
  } from '../jobsListing'
  import { loadWorkspaces, workspace } from '../workspace.svelte'
  import {
    rerunRestartBatchWithConfirm,
    restartBanners,
    type RestartBanner,
  } from '../restartBatches'

  let { scope = 'workspace' }: { scope?: 'workspace' | 'all' } = $props()

  // The filters live in the hash's query, so opening a job and coming back
  // - or sharing the link - keeps the narrowed view
  const statusFilter = hashState('status', '')
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
  const list = resource(
    async () => {
      const [ws, scoped] = [listed, { ...requestScope }]
      const { list: answer, prompts } = await listWithPrompts(ws, scoped, known)
      const rows = answer.jobs.reverse()
      const queued = await queuedEverywhere(rows, !!(ws || scoped.request))
      const progress = await runningProgress(rows)
      return { rows, queueSize: queueLength(rows, queued), prompts, progress }
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
  const visible = $derived(
    jobs.filter(
      (job) =>
        (!statusFilter.value || job.status === statusFilter.value) &&
        jobMatches(job, prompts[job.id] ?? '', nameFilter.value),
    ),
  )
  // Batches a preview found nothing left to rerun in: their banner stays
  // hidden for the rest of the session, in memory only - the client's
  // count is an estimate and the preview corrected it
  let dismissedBatches = $state(new Set<string>())
  // All of a scope's rows, not the name-filtered ones: a batch spans
  // workflows. A status filter that hides cancelled rows hides it too
  const banners = $derived(
    statusFilter.value && statusFilter.value !== 'cancelled'
      ? []
      : restartBanners(jobs, dismissedBatches),
  )

  // Only a click previews: the preview prices every member of the batch,
  // which the 3 s poll must never pay for. Its question and its failures
  // are rerunRestartBatchWithConfirm's
  const rerunning = action(
    async (banner: RestartBanner) => {
      const outcome = await rerunRestartBatchWithConfirm(banner.id)
      if (outcome === 'empty')
        dismissedBatches = new Set([...dismissedBatches, banner.batch])
      return outcome
    },
    { doing: 'rerun the jobs' },
  )

  const moving = action(
    async (id: string, direction: 'up' | 'down' | 'front') => {
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
    <option value="running">running</option>
    <option value="queued">queued</option>
    <option value="succeeded">succeeded</option>
    <option value="failed">failed</option>
    <option value="cancelled">cancelled</option>
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
      disabled={rerunning.busy}
      aria-busy={rerunning.busy}
      title="preview what a rerun would queue, with its cost, before anything is queued"
    >
      Rerun…
    </button>
  </div>
{/each}

<LoadState
  of={list}
  label="jobs"
  empty={!refused && visible.length === 0}
  {filtered}
  onclear={clearFilters}
>
  {#snippet emptyIcon()}<Inbox size={36} strokeWidth={1.5} />{/snippet}
  {#snippet emptyText()}No jobs yet — pick a workflow and run it.{/snippet}
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
          proof={proof && job.workspace === listedGallery ? proof : null}
          onmove={move}
        />
      {/each}
    </div>
  {/if}
</LoadState>

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
