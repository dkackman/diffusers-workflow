<script lang="ts">
  import { Inbox } from 'lucide-svelte'
  import { api } from '../api'
  import type { JobSummary } from '../types'

  let jobs = $state<JobSummary[]>([])
  let error = $state('')

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

  const when = (t: number | null) => (t ? new Date(t * 1000).toLocaleTimeString() : '—')
  const duration = (job: JobSummary) => {
    if (!job.started_at) return ''
    const end = job.finished_at ?? Date.now() / 1000
    return `${(end - job.started_at).toFixed(0)}s`
  }
</script>

<h1>Jobs</h1>

{#if error}<p class="muted">Could not reach the server: {error}</p>{/if}

{#if jobs.length === 0}
  <div class="empty muted">
    <Inbox size={36} strokeWidth={1.5} />
    <p>No jobs yet — pick a workflow and run it.</p>
  </div>
{:else}
  <div class="panel list">
    {#each jobs as job}
      <a
        class="row"
        class:historical={job.historical}
        href={'#/jobs/' + job.id}
        title={job.historical ? 'finished before this server started - loaded from history' : ''}
      >
        <span class="chip {job.status}">{job.status}</span>
        <span class="name">{job.workflow}</span>
        <span class="muted">{when(job.started_at)}</span>
        <span class="muted">{duration(job)}</span>
        <code class="muted">{job.id}</code>
      </a>
    {/each}
  </div>
{/if}

<style>
  h1 { margin-bottom: 1rem; }
  .list { padding: 0.3rem; }
  .row {
    display: grid; grid-template-columns: 90px 1fr auto auto auto;
    gap: 1rem; align-items: center; padding: 0.55rem 0.8rem;
    border-radius: 6px; color: var(--ink);
  }
  .row:hover { background: var(--panel-2); }
  .row.historical { opacity: 0.72; }
  .name { font-weight: 600; }
  .empty {
    display: flex; flex-direction: column; align-items: center; gap: 0.6rem;
    padding: 3rem 0; opacity: 0.8;
  }
</style>
