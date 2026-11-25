<script lang="ts">
  import { gbFromMb } from './format'
  import { jobHref } from './routes'
  import type { WorkerInfo } from './types'
  import { workspace } from './workspace.svelte'

  // One row per card the server runs a worker on - one job per GPU. Shared by
  // the status popover and the server page; `onnavigate` lets the popover
  // close itself when a job link is followed
  let {
    workers,
    onnavigate,
  }: { workers: WorkerInfo[]; onnavigate?: () => void } = $props()

  const state = (w: WorkerInfo) =>
    !w.alive ? 'not started' : w.current_job ? 'running' : 'idle'
</script>

<ul>
  {#each workers as w (w.device)}
    <li>
      <span class="card" title={w.name ?? w.device}>{w.name ?? w.device}</span>
      <span class:muted={!w.current_job}>{state(w)}</span>
      {#if w.pid != null}
        <span class="muted" title="the worker process id">pid {w.pid}</span>
      {/if}
      {#if w.vram_gb != null}
        <span class="muted">{w.vram_gb} GB</span>
      {/if}
      {#if w.host_memory_rss_mb != null}
        <span class="muted" title="resident host memory of the worker process">
          {gbFromMb(w.host_memory_rss_mb)} GB RSS
        </span>
      {/if}
      {#if w.current_job}
        <a
          href={jobHref(w.current_job, w.workspace ?? workspace.current)}
          onclick={() => onnavigate?.()}
        >
          {w.current_job} →
        </a>
      {/if}
    </li>
  {/each}
</ul>

<style>
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: var(--space-2);
  }
  li {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0 var(--space-3);
  }
  .card {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
  }
</style>
