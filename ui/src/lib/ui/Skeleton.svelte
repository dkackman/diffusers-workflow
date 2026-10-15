<script lang="ts">
  // The shape of what is coming, in the panel's own grey, so a slow panel
  // does not jump when it lands. Neutral on purpose: --live is machine
  // state and a load is not one. Still under reduced motion
  let {
    lines = 3,
    tiles = 0,
    label = 'loading',
  }: { lines?: number; tiles?: number; label?: string } = $props()
</script>

<div class="skeleton" role="status" aria-label={label}>
  {#if tiles}
    <div class="tiles">
      {#each [...Array(tiles).keys()] as i (i)}<span class="tile"></span>{/each}
    </div>
  {:else}
    {#each [...Array(lines).keys()] as i (i)}
      <span class="line" style:width="{90 - (i % 3) * 20}%"></span>
    {/each}
  {/if}
</div>

<style>
  .skeleton {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .tiles {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: var(--space-2);
  }
  .tile,
  .line {
    display: block;
    background: var(--panel-2);
    border-radius: var(--radius-2);
    animation: pulse 1.4s ease-in-out infinite;
  }
  .tile {
    aspect-ratio: 1;
  }
  .line {
    height: 0.9em;
  }
  @keyframes pulse {
    50% {
      opacity: 0.55;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .tile,
    .line {
      animation: none;
    }
  }
</style>
