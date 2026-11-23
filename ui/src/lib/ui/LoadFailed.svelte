<script lang="ts">
  // In place of an editor whose document never arrived: what failed, a
  // Retry, and - when the file does not exist - a way to start it. Phase 2's
  // LoadState absorbs it
  import { describeError } from '../apiErrors'

  let {
    label,
    error,
    onretry,
    oncreate,
  }: {
    label: string
    error: unknown
    onretry: () => void
    /** Offered only when the file is missing, not the workspace */
    oncreate?: () => void
  } = $props()

  const described = $derived(describeError(error))
</script>

<div class="panel loadfailed" role="alert">
  <p>Couldn't open {label}: {described.title}</p>
  {#if described.remedy}<p>{described.remedy}</p>{/if}
  <div class="actions">
    <button onclick={onretry}>Retry</button>
    {#if oncreate && described.kind === 'not_found'}
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
