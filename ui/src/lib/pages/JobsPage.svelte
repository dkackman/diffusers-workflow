<script lang="ts">
  import { ChevronDown, ChevronUp, ChevronsUp, Inbox } from '@lucide/svelte'
  import { api } from '../api'
  import { notify } from '../toast'
  import type { JobSummary } from '../types'

  let jobs = $state<JobSummary[]>([])
  let error = $state('')
  let statusFilter = $state('')
  let nameFilter = $state('')

  const visible = $derived(
    jobs.filter(
      (job) =>
        (!statusFilter || job.status === statusFilter) &&
        job.workflow.toLowerCase().includes(nameFilter.toLowerCase()),
    ),
  )

  $effect(() => {
    const poll = async () => {
      try {
        jobs = (await api.listJobs()).jobs.reverse()
        error = ''
      } catch (e) {
        error = e instanceof Error ? e.message : String(e)
      }
    }
    poll()
    const timer = setInterval(poll, 3000)
    return () => clearInterval(timer)
  })

  const queuedCount = $derived(
    jobs.filter((job) => job.queue_position !== undefined).length,
  )

  async function move(
    event: MouseEvent,
    id: string,
    direction: 'up' | 'down' | 'front',
  ) {
    event.preventDefault()
    try {
      await api.moveJob(id, direction)
      jobs = (await api.listJobs()).jobs.reverse()
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      notify.error(msg)
    }
  }

  const when = (t: number | null) =>
    t ? new Date(t * 1000).toLocaleTimeString() : '—'
  const duration = (job: JobSummary) => {
    if (!job.started_at) return ''
    const end = job.finished_at ?? Date.now() / 1000
    return `${(end - job.started_at).toFixed(0)}s`
  }
</script>

<div class="head">
  <h1>Jobs</h1>
  <select bind:value={statusFilter} title="filter by status">
    <option value="">all statuses</option>
    <option value="running">running</option>
    <option value="queued">queued</option>
    <option value="succeeded">succeeded</option>
    <option value="failed">failed</option>
    <option value="cancelled">cancelled</option>
  </select>
  <input
    class="filter"
    placeholder="filter by workflow…"
    bind:value={nameFilter}
  />
</div>

{#if error}<p class="muted">Could not reach the server: {error}</p>{/if}

{#if jobs.length === 0}
  <div class="empty muted">
    <Inbox size={36} strokeWidth={1.5} />
    <p>No jobs yet — pick a workflow and run it.</p>
  </div>
{:else if visible.length === 0}
  <p class="muted">No jobs match the filter.</p>
{:else}
  <div class="panel list">
    {#each visible as job (job.id)}
      <a
        class="row"
        class:historical={job.historical}
        href={'#/jobs/' + job.id}
        title={job.historical
          ? 'finished before this server started - loaded from history'
          : ''}
      >
        <span class="chip {job.status}">{job.status}</span>
        <span class="name">
          {job.workflow}
          {#if job.queue_position !== undefined}
            <span class="qpos" title="position in the waiting queue"
              >#{job.queue_position + 1}</span
            >
            {#if queuedCount > 1}
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
                {#if job.queue_position < queuedCount - 1}
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
        <span class="muted started">{when(job.started_at)}</span>
        <span class="muted dur">{duration(job)}</span>
        <code class="muted jobid">{job.id}</code>
      </a>
    {/each}
  </div>
{/if}

<style>
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.8rem;
    margin-bottom: 1rem;
  }
  .head h1 {
    margin: 0;
    flex: 1;
  }
  .head select {
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
    .started,
    .dur {
      font-size: 0.85rem;
    }
  }
  .row:hover {
    background: var(--panel-2);
  }
  .row.historical {
    opacity: 0.72;
  }
  .name {
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem;
    min-width: 0;
    overflow-wrap: anywhere;
  }
  .qpos {
    font-size: 0.72rem;
    color: var(--muted);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 0 0.3rem;
    font-variant-numeric: tabular-nums;
  }
  .qmove {
    display: inline-flex;
    gap: 0.1rem;
  }
  .icon {
    display: inline-flex;
    padding: 0.15rem 0.3rem;
  }
  .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.6rem;
    padding: 3rem 0;
    opacity: 0.8;
  }
</style>
