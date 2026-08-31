<script lang="ts">
  import type { Snippet } from 'svelte'

  let { icon, children }: { icon: Snippet; children: Snippet } = $props()
</script>

<div class="empty muted">
  {@render icon()}
  <p>{@render children()}</p>
</div>

<style>
  .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-5) 0;
    text-align: center;
  }
</style>
