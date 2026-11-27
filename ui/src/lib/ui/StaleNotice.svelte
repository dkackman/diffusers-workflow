<script lang="ts">
  import { TriangleAlert } from '@lucide/svelte'

  // A write the server refused because what it holds moved since the page
  // read it (a 412 or 409 `stale_write`): Reload drops the edit for the
  // server's copy, Overwrite writes it over that copy anyway
  let {
    what,
    onreload,
    onoverwrite,
  }: { what: string; onreload: () => void; onoverwrite: () => void } = $props()
</script>

<div class="panel warn-edge notice" role="alert">
  <p>
    <TriangleAlert size={14} aria-hidden="true" />
    <span>
      {what} changed since you opened it, in another tab or by an agent.
    </span>
  </p>
  <div class="answers">
    <button type="button" class="quiet" onclick={onreload}>Reload</button>
    <button type="button" class="quiet danger" onclick={onoverwrite}>
      Overwrite
    </button>
  </div>
</div>

<style>
  .notice {
    flex-basis: 100%;
    padding: 0.55rem 0.8rem;
    font-size: var(--t-sm);
    display: grid;
    gap: var(--space-2);
  }
  /* the icon beside the sentence, the two answers in a row under it */
  p,
  .answers {
    margin: 0;
    display: flex;
    gap: var(--space-2);
  }
  p > :global(svg) {
    margin-top: 0.25em;
    flex-shrink: 0;
  }
</style>
