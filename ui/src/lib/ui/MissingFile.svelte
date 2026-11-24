<script lang="ts">
  // What a page says under its header when the link's open= names a file
  // its view does not list (mediaViewer.ts): the leaf, until dismissed
  import { X } from '@lucide/svelte'

  let { leaf = $bindable('') }: { leaf: string } = $props()
</script>

{#if leaf}
  <p class="hint missing">
    {leaf} is not in this view any more.
    <button
      type="button"
      class="quiet icon"
      aria-label="Dismiss"
      title="Dismiss"
      onclick={() => (leaf = '')}
    >
      <X size={14} />
    </button>
  </p>
{/if}

<style>
  .missing {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin: 0 0 var(--space-2);
  }
</style>
