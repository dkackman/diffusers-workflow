<script lang="ts">
  import type { Rung } from '../preferences'

  // Under the tier dropdown (#1762 stage B): for each tier, the workflow
  // this host would run, the targets it passed over greyed with their
  // reason, or why no target fits. The tier the form holds is marked. The
  // figure is the run check's, for the chosen tier only
  let { key, rungs, current }: { key: string; rungs: Rung[]; current: string } =
    $props()
</script>

<ul class="rungs" data-rungs-for={key} aria-label="what each {key} runs here">
  {#each rungs as rung (rung.tier)}
    <li
      data-tier={rung.tier}
      class:current={rung.tier === current}
      aria-current={rung.tier === current ? 'true' : undefined}
    >
      <span class="tier">{rung.tier}</span>
      <span class="pick">
        {#if rung.target}
          <code>{rung.target}</code>
        {:else}
          <span class="warn">{rung.error ?? 'nothing fits this host'}</span>
        {/if}
        {#each rung.skipped as skip (skip.workflow)}
          <span class="skip muted" title={skip.reason}>
            <code>{skip.workflow}</code> skipped: {skip.reason}
          </span>
        {/each}
      </span>
    </li>
  {/each}
</ul>

<style>
  /* Under the field, in the field's column, as VariableNotes' hints */
  .rungs {
    grid-column: 2 / -1;
    list-style: none;
    margin: calc(-1 * var(--space-1)) 0 0;
    padding: 0;
    font-size: var(--t-sm);
  }
  /* The tier in its own column, so a long refusal wraps beside it */
  li {
    display: grid;
    grid-template-columns: 5.5em 1fr;
    align-items: baseline;
    gap: var(--space-2);
  }
  .tier {
    color: var(--muted);
  }
  .pick {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: var(--space-1) var(--space-2);
  }
  .current .tier {
    color: inherit;
    font-weight: 600;
  }
  .skip code {
    text-decoration: line-through;
  }
  .warn {
    color: var(--warn);
  }
</style>
