<script lang="ts">
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
  <p class="muted">No jobs yet — pick a workflow and run it.</p>
{:else}
  <div class="panel list">
    {#each jobs as job}
      <a class="row" href={'#/jobs/' + job.id}>
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
  .name { font-weight: 600; }
</style>
