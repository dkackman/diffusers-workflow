<script lang="ts">
  import type { LevelFinding } from '../types'
  import ErrorLine from '../ui/ErrorLine.svelte'
  import {
    formatAt,
    measuredLine,
    warnFirst,
    type AssessState,
  } from './assess.svelte'

  // The detail panel's Findings: the level findings the metadata reply
  // carries, until an Assess has answered - then that reply's list, which
  // holds those rules too, so nothing is counted twice
  let {
    findings = [],
    assessed,
  }: {
    findings?: LevelFinding[]
    assessed?: AssessState
  } = $props()

  const rows = $derived(warnFirst(assessed?.reply?.findings ?? findings))
</script>

<section class="findings" aria-label="Findings">
  <span class="muted">Findings</span>
  {#if rows.length}
    <ul>
      {#each rows as row, i (i)}
        <li>
          <span class="severity sev-{row.severity}">{row.severity}</span>
          <span class="says">{row.says}</span>
          {#if formatAt(row.at)}
            <code class="at">{formatAt(row.at)}</code>
          {/if}
        </li>
      {/each}
    </ul>
  {:else}
    <div class="none">No findings</div>
  {/if}
  {#if assessed?.reply}
    <div class="measured muted">{measuredLine(assessed.reply)}</div>
  {/if}
  {#if assessed && assessed.error != null}
    <ErrorLine error={assessed.error} doing="assess this file" />
  {/if}
</section>

<style>
  ul {
    list-style: none;
    margin: 0.15rem 0 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }
  li {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.4rem;
  }
  .severity {
    font-family: var(--font-mono);
    font-size: var(--t-xs);
    padding: 0 0.35rem;
    border: 1px solid currentColor;
    border-radius: var(--radius-1);
  }
  /* Their own tokens (app.css): neither machine state nor the user's place */
  .sev-warn {
    color: var(--finding-warn);
  }
  .sev-info {
    color: var(--finding-info);
  }
  .says {
    flex: 1 1 20ch;
  }
  .at {
    font-size: var(--t-xs);
  }
  .measured {
    margin-top: 0.3rem;
    font-size: var(--t-xs);
    overflow-wrap: anywhere;
  }
</style>
