<script lang="ts">
  import { cardReadings } from './cardMemory'
  import CardMemoryList from './CardMemoryList.svelte'
  import { connection } from './connection.svelte'
  import WorkerList from './WorkerList.svelte'

  // The server status rows, one set for the header's popover and the Server
  // page. They read `connection` (connection.svelte.ts), so the two cannot
  // disagree: the worker row speaks only when a health reading exists
  let { onnavigate }: { onnavigate?: () => void } = $props()

  const health = $derived(connection.health)
  const cards = $derived(cardReadings(connection.memory))
  const silent = $derived(
    connection.state === 'offline' || connection.state === 'locked',
  )
</script>

<dl class="kv">
  <dt>Server</dt>
  <dd>
    {#if connection.state === 'offline'}
      <span class="bad">unreachable</span>
    {:else if connection.state === 'locked'}
      <span class="warn">needs the API token</span>
    {:else if health}
      {health.status}{#if health.version}&nbsp;· v{health.version}{/if}
    {:else}
      <span class="muted">checking…</span>
    {/if}
  </dd>

  <dt>{(health?.workers.length ?? 0) > 1 ? 'Workers' : 'Worker'}</dt>
  <dd>
    {#if health?.workers.length}
      <WorkerList workers={health.workers} {onnavigate} />
    {/if}
    {#if health && !health.worker_alive && !silent}
      <span class="muted">not started — spawns with the first job</span>
    {/if}
  </dd>

  <dt>Queue</dt>
  <dd>{health?.queued ?? 0} queued</dd>

  <dt>Memory</dt>
  <dd><CardMemoryList {cards} /></dd>
</dl>

<style>
</style>
