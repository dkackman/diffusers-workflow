<script lang="ts">
  import { gbFromMb } from './format'
  import type { HealthInfo, MemoryInfo } from './types'
  import Popover from './ui/Popover.svelte'

  let {
    open = $bindable(false),
    anchor = null,
    health,
    memory,
  }: {
    open?: boolean
    anchor?: HTMLElement | null
    health: HealthInfo | null
    memory: MemoryInfo | null
  } = $props()

  const gb = gbFromMb
  const info = $derived(memory?.info ?? null)
</script>

<Popover bind:open label="server status" {anchor}>
  <dl>
    <dt>Server</dt>
    <dd>
      {#if health}
        {health.status}{#if health.version}&nbsp;· v{health.version}{/if}
      {:else}
        <span class="bad">unreachable</span>
      {/if}
    </dd>

    <dt>Worker</dt>
    <dd>
      {#if health?.worker_alive}
        running
      {:else}
        <span class="muted">not started — spawns with the first job</span>
      {/if}
    </dd>

    <dt>Queue</dt>
    <dd>{health?.queued ?? 0} queued</dd>

    {#if health?.current_job}
      <dt>Job</dt>
      <dd>
        <a href={'#/jobs/' + health.current_job} onclick={() => (open = false)}
          >watch the running job →</a
        >
      </dd>
    {/if}

    <dt>Memory</dt>
    <dd>
      {#if info?.gpu_available}
        {info.gpu_device_name} · {gb(info.gpu_memory_allocated_mb ?? 0)} GB allocated
        {#if info.gpu_memory_total_mb}
          of {gb(info.gpu_memory_total_mb)} GB
        {:else}
          <span class="muted">(this backend reports allocated only)</span>
        {/if}
      {:else}
        <span class="muted">no reading yet</span>
      {/if}
    </dd>
  </dl>
</Popover>

<style>
  dl {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: var(--space-2) var(--space-4);
    margin: 0;
    align-items: baseline;
  }
  dt {
    font-weight: 600;
    color: var(--muted);
    font-size: 0.75rem;
    text-transform: none;
  }
  dd {
    margin: 0;
  }
  .bad {
    color: var(--bad);
  }
</style>
