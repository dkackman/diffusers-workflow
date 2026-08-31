<script lang="ts">
  import type { Snippet } from 'svelte'
  import { X } from '@lucide/svelte'
  import { storageGet, storageSet } from './storage'

  let { storageKey, children }: { storageKey: string; children: Snippet } =
    $props()

  // The legacy pages stored the string '1' - any truthy stored value
  // counts as dismissed
  // svelte-ignore state_referenced_locally
  let show = $state(!storageGet(storageKey, false))

  function dismiss() {
    show = false
    storageSet(storageKey, true)
  }
</script>

{#if show}
  <div class="hintbar muted">
    <span>{@render children()}</span>
    <button
      class="quiet icon"
      onclick={dismiss}
      title="dismiss"
      aria-label="dismiss this hint"
    >
      <X size={13} />
    </button>
  </div>
{/if}

<style>
  .hintbar {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    border: 1px dashed var(--line);
    border-radius: var(--radius-1);
    padding: 0.45rem 0.7rem;
    font-size: 0.85rem;
    margin-bottom: var(--space-4);
  }
  .hintbar span {
    flex: 1;
  }
  .hintbar .icon {
    display: inline-flex;
    padding: 0.2rem 0.3rem;
  }
</style>
