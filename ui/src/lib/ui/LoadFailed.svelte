<script lang="ts">
  // In place of an editor whose document never arrived: what failed, a
  // Retry, and - when the file does not exist - a way to start it. Phase 2's
  // LoadState absorbs it
  import { ApiError } from '../apiErrors'

  let {
    label,
    error,
    onretry,
    oncreate,
  }: {
    label: string
    error: unknown
    onretry: () => void
    /** Offered only when the file is missing (a 404) */
    oncreate?: () => void
  } = $props()

  const reason = $derived(
    error instanceof ApiError && error.status === 404
      ? 'not found'
      : error instanceof Error
        ? error.message
        : String(error),
  )
</script>

<div class="panel loadfailed" role="alert">
  <p>Couldn't open {label}: {reason}</p>
  <div class="actions">
    <button onclick={onretry}>Retry</button>
    {#if oncreate}
      <button class="quiet" onclick={oncreate}>
        Start a new one named {label}
      </button>
    {/if}
  </div>
</div>

<style>
  .loadfailed {
    margin: 1rem 0;
  }
  .loadfailed p {
    margin: 0 0 0.6rem;
    overflow-wrap: anywhere;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
</style>
