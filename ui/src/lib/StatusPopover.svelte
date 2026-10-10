<script lang="ts">
  import { cardReadings } from './cardMemory'
  import CardMemoryList from './CardMemoryList.svelte'
  import type { HealthInfo, MemoryInfo } from './types'
  import Popover from './ui/Popover.svelte'
  import WorkerList from './WorkerList.svelte'

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

  const cards = $derived(cardReadings(memory))
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

    <dt>{(health?.workers.length ?? 0) > 1 ? 'Workers' : 'Worker'}</dt>
    <dd>
      {#if health?.workers.length}
        <WorkerList
          workers={health.workers}
          onnavigate={() => (open = false)}
        />
      {/if}
      {#if !health?.worker_alive}
        <span class="muted">not started — spawns with the first job</span>
      {/if}
    </dd>

    <dt>Queue</dt>
    <dd>{health?.queued ?? 0} queued</dd>

    <dt>Memory</dt>
    <dd><CardMemoryList {cards} /></dd>
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
