<script lang="ts">
  import { formatDuration, formatExact, formatMtime } from '../format'
  import { holdKind } from '../progress'
  import { queueWait } from '../jobTimes'
  import { resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import { onMount } from 'svelte'
  import { ChevronDown, ChevronUp, ChevronsUp, Inbox, X } from '@lucide/svelte'
  import { api } from '../api'
  import { notify } from '../toast'
  import type { JobSummary } from '../types'
  import { wsHref } from '../routes'
  import { hashQuery, setHashQuery } from '../router.svelte'
  import { parseRequestFilter, type RequestFilter } from '../requestFilter'
  import { listingRefused, listingWorkspace, queueLength } from '../jobsListing'
  import { loadWorkspaces, workspace } from '../workspace.svelte'
  import {
    rerunRestartBatchWithConfirm,
    restartBanners,
    type RestartBanner,
  } from '../restartBatches'

  let { scope = 'workspace' }: { scope?: 'workspace' | 'all' } = $props()

  // The filters live in the hash's query, so opening a job and coming back
  // - or sharing the link - keeps the narrowed view
  const initial = hashQuery()
  let statusFilter = $state(initial.status ?? '')
  let nameFilter = $state(initial.q ?? '')
  // Server-side, unlike the two filters above: it changes which jobs are
  // fetched at all, not just which of the fetched ones are shown. Only read
  // under scope 'all' - empty there means every workspace, which is the
  // point of that view; a workspace's own Jobs page filters by
  // `workspace.current` instead.
  let workspaceFilter = $state(initial.ws ?? '')
  // Server-side too: `req=<id>` or `req=<id>/<item>` asks the server for
  // the jobs linked to that requests-ledger request (or one item of it)
  let reqFilter = $state(initial.req ?? '')
  const requestScope = $derived(parseRequestFilter(reqFilter))

  $effect(() =>
    setHashQuery({
      status: statusFilter,
      q: nameFilter,
      ws: scope === 'all' ? workspaceFilter : '',
      req: reqFilter,
    }),
  )
  // The route's view is unchanged when the Jobs nav link lands on the bare
  // route from a narrowed one, so the page is not re-created: the filters
  // follow the hash too, and a hash with no query clears them
  $effect(() => {
    const sync = () => {
      const query = hashQuery()
      statusFilter = query.status ?? ''
      nameFilter = query.q ?? ''
      workspaceFilter = query.ws ?? ''
      reqFilter = query.req ?? ''
    }
    window.addEventListener('hashchange', sync)
    return () => window.removeEventListener('hashchange', sync)
  })
  // The server-side filters narrow what is fetched, so an empty answer is
  // "nothing matches"; the client-side ones only mean that when there were
  // rows to filter (on an empty listing the answer is "no jobs yet")
  const narrowed = $derived(
    !!requestScope.request || (scope === 'all' && !!workspaceFilter),
  )
  function clearFilters() {
    statusFilter = ''
    nameFilter = ''
    workspaceFilter = ''
    reqFilter = ''
  }
  // A request lives in one workspace, so its filter is never read across
  // them (listingWorkspace): on the all-jobs view it pins the picker to the
  // workspace it is read in, and choosing every workspace lets go of it
  const listed = $derived(
    listingWorkspace(
      scope,
      workspaceFilter,
      workspace.current,
      requestScope.request,
    ),
  )
  $effect(() => {
    if (scope === 'all' && listed !== workspaceFilter) workspaceFilter = listed
  })

  // Only worth a picker once there is a choice - a single-workspace server
  // has nothing to filter by
  onMount(loadWorkspaces)

  // Live state, as it was: the listing the filters name, read every 3 s.
  // What load reads before its first await (the scope's workspace and the
  // request filter) restarts the read when it changes, so a narrowed view
  // never shows another question's rows
  const list = resource(
    async () => {
      const [ws, scoped] = [listed, requestScope]
      const rows = (await listJobs(ws, scoped)).jobs.reverse()
      const queued = await queuedEverywhere(rows, !!(ws || scoped.request))
      return { rows, queueSize: queueLength(rows, queued) }
    },
    { live: { everyMs: 3000 } },
  )
  // The server turning the question down: its message is the explanation,
  // and the rows of an earlier question must not stay under this one's
  const refused = $derived(
    list.status === 'error' && listingRefused(list.error),
  )
  const jobs = $derived(refused ? [] : (list.data?.rows ?? []))
  const filtered = $derived(
    narrowed || ((!!statusFilter || !!nameFilter) && jobs.length > 0),
  )
  const queueSize = $derived(list.data?.queueSize ?? 0)
  const visible = $derived(
    jobs.filter(
      (job) =>
        (!statusFilter || job.status === statusFilter) &&
        (job.workflow ?? '').toLowerCase().includes(nameFilter.toLowerCase()),
    ),
  )
  const listJobs = (ws: string, scoped: RequestFilter) =>
    api.listJobs(
      ws || undefined,
      undefined,
      undefined,
      scoped.request || undefined,
      scoped.item || undefined,
    )
  /** The server's count of every queued job, asked for only by a narrowed
   * listing that shows a waiting one: an unnarrowed listing holds the whole
   * queue, and 0 leaves the length to the rows. */
  async function queuedEverywhere(rows: JobSummary[], narrowed: boolean) {
    if (!narrowed || !rows.some((job) => job.queue_position !== undefined))
      return 0
    return (await api.listJobs(undefined, 1, 'queued')).total
  }

  // Batches a preview found nothing left to rerun in: their banner stays
  // hidden for the rest of the session, in memory only - the client's
  // count is an estimate and the preview corrected it
  let dismissedBatches = $state(new Set<string>())
  let previewing = $state('')
  // All of a scope's rows, not the name-filtered ones: a batch spans
  // workflows. A status filter that hides cancelled rows hides it too
  const banners = $derived(
    statusFilter && statusFilter !== 'cancelled'
      ? []
      : restartBanners(jobs, dismissedBatches),
  )

  // Only a click previews: the preview prices every member of the batch,
  // which the 3 s poll must never pay for
  async function rerunBatch(banner: RestartBanner) {
    previewing = banner.batch
    try {
      const outcome = await rerunRestartBatchWithConfirm(banner.id)
      if (outcome === 'empty')
        dismissedBatches = new Set([...dismissedBatches, banner.batch])
    } finally {
      previewing = ''
    }
  }

  async function move(
    event: MouseEvent,
    id: string,
    direction: 'up' | 'down' | 'front',
  ) {
    event.preventDefault()
    try {
      await api.moveJob(id, direction)
    } catch (e) {
      notify.failure('move the job', e)
      return
    }
    // Same listing the poll asks for - refreshing without its filters would
    // flash every workspace's jobs into a view the user had narrowed to one
    await list.reload()
  }

  const requestLabel = (f: { request: string; item: string }) =>
    f.item ? `${f.request} › ${f.item}` : f.request
  const when = (t: number | null) => (t ? formatMtime(t) : '—')
  const waitOf = (job: JobSummary) => queueWait(job, Date.now() / 1000)
  // The row shows the compact start; hovering gives it to the second, with
  // the finish and any time spent queued
  const timesOf = (job: JobSummary) => {
    const wait = waitOf(job)
    return [
      job.started_at && `started ${formatExact(job.started_at)}`,
      job.finished_at && `finished ${formatExact(job.finished_at)}`,
      wait !== null &&
        wait >= 1 &&
        `waited ${formatDuration(wait)} in the queue`,
    ]
      .filter(Boolean)
      .join('\n')
  }
  const duration = (job: JobSummary) => {
    if (!job.started_at) return ''
    const end = job.finished_at ?? Date.now() / 1000
    return formatDuration(end - job.started_at)
  }
</script>

<div class="pagehead">
  <h1>{scope === 'all' ? 'All jobs' : 'Jobs'}</h1>
  <select bind:value={statusFilter} title="filter by status">
    <option value="">all statuses</option>
    <option value="running">running</option>
    <option value="queued">queued</option>
    <option value="succeeded">succeeded</option>
    <option value="failed">failed</option>
    <option value="cancelled">cancelled</option>
  </select>
  {#if scope === 'all' && (workspace.names?.length ?? 0) > 1}
    <select
      bind:value={workspaceFilter}
      onchange={() => {
        if (!workspaceFilter) reqFilter = ''
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
    placeholder="filter by workflow…"
    bind:value={nameFilter}
  />
  {#if requestScope.request}
    <span class="wschip active-filter" data-testid="req-filter">
      <a
        class="plain"
        href={wsHref(listed, 'requests', requestScope.request)}
        title="open this request"
      >
        {requestLabel(requestScope)}
      </a>
      <button
        class="quiet icon"
        onclick={() => (reqFilter = '')}
        title="remove the request filter"
        aria-label="remove the request filter"
      >
        <X size={12} />
      </button>
    </span>
  {/if}
</div>

{#each banners as banner (banner.batch)}
  <div class="panel restart" role="status" data-batch={banner.batch}>
    <span>
      {`${banner.count} ${banner.count === 1 ? 'job was' : 'jobs were'} cancelled by a server restart`}
    </span>
    <button
      onclick={() => rerunBatch(banner)}
      disabled={previewing === banner.batch}
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
        <a
          class="row plain"
          class:historical={job.historical}
          class:row-live={job.status === 'running'}
          href={wsHref(job.workspace, 'jobs', job.id)}
          title={job.historical
            ? 'finished before this server started - loaded from history'
            : ''}
        >
          <span class="chip {job.status}">{job.status}</span>
          <span class="namecol">
            <span class="name">
              {job.workflow}
              {#if job.run_version}
                <span
                  class="wschip muted"
                  title="version {job.run_version} of this workflow - the gallery labels its files the same way"
                >
                  v{job.run_version}
                </span>
              {/if}
              {#if job.rerun_of}
                <span class="wschip muted" title="a rerun of {job.rerun_of}">
                  rerun
                </span>
              {/if}
              {#if job.request}
                <span
                  class="wschip muted"
                  title="request {job.request}{job.item
                    ? `, item ${job.item}`
                    : ''}"
                >
                  {requestLabel({
                    request: job.request,
                    item: job.item ?? '',
                  })}
                </span>
              {/if}
              {#if scope === 'all' && (workspace.names?.length ?? 0) > 1}
                <span class="wschip muted" title="workspace">
                  {job.workspace}
                </span>
              {/if}
              {#if job.device}
                <span class="wschip muted" title="the card this job ran on">
                  {job.device}
                </span>
              {/if}
              {#if job.acknowledged === 'bound'}
                <span
                  class="wschip muted"
                  title="queued with a cost acknowledgement bound to the validated plan"
                >
                  acknowledged
                </span>
              {/if}
              {#if job.queue_position !== undefined}
                <span class="qpos" title="position in the waiting queue">
                  #{job.queue_position + 1}
                </span>
                {#if queueSize > 1}
                  <span class="qmove">
                    {#if job.queue_position > 0}
                      <button
                        class="quiet icon"
                        onclick={(e) => move(e, job.id, 'front')}
                        title="run this job next"
                        aria-label="move {job.id} to the front of the queue"
                      >
                        <ChevronsUp size={13} />
                      </button>
                      <button
                        class="quiet icon"
                        onclick={(e) => move(e, job.id, 'up')}
                        title="move up one place"
                        aria-label="move {job.id} up one place"
                      >
                        <ChevronUp size={13} />
                      </button>
                    {/if}
                    {#if job.queue_position < queueSize - 1}
                      <button
                        class="quiet icon"
                        onclick={(e) => move(e, job.id, 'down')}
                        title="move down one place"
                        aria-label="move {job.id} down one place"
                      >
                        <ChevronDown size={13} />
                      </button>
                    {/if}
                  </span>
                {/if}
              {/if}
            </span>
            {#if job.reason}
              <span class="reason muted" title={job.reason}>{job.reason}</span>
            {/if}
          </span>
          {#if job.status === 'queued'}
            {@const wait = waitOf(job)}
            <span
              class="muted started"
              title={job.created_at
                ? `queued ${formatExact(job.created_at)} - time spent waiting for a card`
                : 'time spent waiting for a card'}
            >
              waiting {wait === null ? '' : formatDuration(wait)}
            </span>
            {#if job.waiting_for}
              {@const kind = holdKind(job.waiting_for)}
              <span class="muted" title={job.waiting_for}>
                · {kind === 'host_ram'
                  ? 'host RAM'
                  : kind === 'cards_busy'
                    ? 'cards busy'
                    : 'held'}
              </span>
            {/if}
          {:else}
            <span class="muted started" title={timesOf(job)}>
              {when(job.started_at)}
            </span>
          {/if}
          <span class="muted dur">{duration(job)}</span>
          <code class="muted jobid">{job.id}</code>
        </a>
      {/each}
    </div>
  {/if}
</LoadState>

<style>
  .pagehead h1 {
    margin: 0;
    flex: 1;
  }
  .pagehead select {
    max-width: 150px;
  }
  .filter {
    max-width: 220px;
  }
  .list {
    padding: 0.3rem;
  }
  .row {
    display: grid;
    grid-template-columns: 90px minmax(0, 1fr) auto auto auto;
    gap: 1rem;
    align-items: center;
    padding: 0.55rem 0.8rem;
    border-radius: 6px;
    color: var(--ink);
  }
  /* The job id is the widest unbreakable cell and is repeated on the job
     page itself, so it is the first thing to go as the row narrows; below
     that the timestamps drop to a second line under the name */
  @media (max-width: 860px) {
    .row {
      grid-template-columns: 90px minmax(0, 1fr) auto auto;
    }
    .jobid {
      display: none;
    }
  }
  @media (max-width: 560px) {
    .row {
      grid-template-columns: auto minmax(0, 1fr);
      gap: 0.2rem 0.7rem;
    }
  }
  .row:hover {
    background: var(--panel-2);
  }
  .row.historical {
    opacity: 0.72;
  }
  /* The name and, under it, why a failed or cancelled job ended - each its
     own element, so a later line beside the reason is a sibling */
  .namecol {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    min-width: 0;
  }
  .reason {
    font-size: var(--t-xs);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
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
  /* A workflow name is what the engine resolved to run this job */
  .name {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem;
    min-width: 0;
    overflow-wrap: anywhere;
  }
  .started,
  .dur {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: var(--t-xs);
  }
  .qpos {
    font-family: var(--font-mono);
    font-size: var(--t-xs);
    color: var(--muted);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 0 0.3rem;
    font-variant-numeric: tabular-nums;
  }
  .wschip {
    font-family: var(--font-mono);
    font-size: var(--t-xs);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 0 0.3rem;
  }
  .active-filter {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
  }
  .qmove {
    display: inline-flex;
    gap: 0.1rem;
  }
  .icon {
    display: inline-flex;
    padding: 0.15rem 0.3rem;
  }
</style>
