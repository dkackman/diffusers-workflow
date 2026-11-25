<script lang="ts">
  import type { Finding } from '../findings'

  let {
    key,
    mode,
    selector = undefined,
    hints = undefined,
    findings = undefined,
  }: {
    key: string
    mode: 'define' | 'override'
    /** The presets this variable selects among (`presetView`) */
    selector?: { bundles: string[]; sets: string[] }
    /** Its declared rules in words, and what a validate answer said */
    hints?: string[]
    findings?: Finding[]
  } = $props()
</script>

{#if selector}
  <ul class="hints" data-presets-for={key}>
    {#if selector.sets.length}
      <li>sets {selector.sets.join(', ')}</li>
    {/if}
    {#if mode === 'define' && selector.bundles.length}
      <li>bundles: {selector.bundles.join(', ')}</li>
    {/if}
  </ul>
{/if}
{#if hints?.length}
  <ul class="hints" data-hints-for={key}>
    {#each hints as hint, i (i)}
      <li>{hint}</li>
    {/each}
  </ul>
{/if}
{#if findings?.length}
  <ul class="hints findings" data-findings-for={key}>
    {#each findings as finding, i (i)}
      <li class={finding.tone}>{finding.text}</li>
    {/each}
  </ul>
{/if}

<style>
  /* Under the field, in the field's column; spans the row when narrow */
  .hints {
    grid-column: 2 / -1;
    margin: calc(-1 * var(--space-1)) 0 0;
    padding: 0;
    list-style: none;
    font-size: 0.85em;
    color: var(--muted);
    overflow-wrap: anywhere;
  }
  @container (max-width: 400px) {
    .hints {
      grid-column: 1 / -1;
    }
  }
</style>
