<script lang="ts">
  import type { HealthInfo, MemoryInfo } from './types'

  let {
    open = $bindable(false),
    health,
    memory,
  }: {
    open?: boolean
    health: HealthInfo | null
    memory: MemoryInfo | null
  } = $props()

  const gb = (mb: number) => (mb / 1024).toFixed(1)
  const info = $derived(memory?.info ?? null)

  // Click-anywhere-else closes; the toggle button and the panel itself
  // stop propagation so their clicks never reach this handler
  function onWindowClick() {
    if (open) open = false
  }
</script>

<svelte:window onclick={onWindowClick} />

{#if open}
  <!-- The click handler only stops propagation so inside-clicks don't hit
       the window's close-on-click - there is nothing to mirror on keyboard -->
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="pop panel"
    role="dialog"
    aria-label="server status"
    tabindex="-1"
    onclick={(e) => e.stopPropagation()}
  >
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
          <a
            href={'#/jobs/' + health.current_job}
            onclick={() => (open = false)}>watch the running job →</a
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
  </div>
{/if}

<style>
  .pop {
    position: absolute;
    top: calc(100% + 4px);
    left: var(--space-4);
    z-index: 30;
    min-width: min(320px, 90vw);
    box-shadow: 0 6px 24px color-mix(in srgb, var(--bg) 60%, transparent);
    font-size: 0.85rem;
  }
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
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  dd {
    margin: 0;
  }
  .bad {
    color: var(--bad);
  }
</style>
