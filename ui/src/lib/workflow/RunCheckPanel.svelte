<script lang="ts">
  // What validate says a Run with these values would do, checked as you
  // edit. It informs; it never gates Run. Every finding is server text,
  // shown as text
  import type { Finding } from '../findings'
  import ErrorLine from '../ui/ErrorLine.svelte'
  import type { PlanLine } from '../plan'
  import type { Verdict } from '../runCheck.svelte'

  let {
    verdict,
    checking,
    planLines,
    checkError,
    findings,
  }: {
    verdict: Verdict | null
    checking: boolean
    planLines: PlanLine[]
    /** A check the server refused outright, as caught, first in the list */
    checkError: unknown
    /** What validate said that names no field the form shows */
    findings: Finding[]
  } = $props()
</script>

<section class="panel check" aria-label="run check" aria-busy={checking}>
  <div class="verdict">
    {#if verdict}
      <span class={verdict.tone}>{verdict.text}</span>
    {/if}
    {#if checking}<span class="muted">checking…</span>{/if}
  </div>
  {#if planLines.length}
    <ul
      class="plan"
      aria-label="what a run will do"
      title="from the workflow's own cost block and this server's model cache - a plan, not a promise"
    >
      {#each planLines as line, i (i)}
        <li class:warn={line.tone === 'warn'}>{line.text}</li>
      {/each}
    </ul>
  {/if}
  {#if checkError != null || findings.length}
    <ul class="findings" aria-label="findings">
      {#if checkError != null}
        <li class="bad">
          <ErrorLine error={checkError} doing="check these values" />
        </li>
      {/if}
      {#each findings as finding, i (i)}
        <li class={finding.tone}>{finding.text}</li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  /* The check beside Run: the verdict, then the plan in mono (figures the
     engine resolves), then what validate said that names no field */
  .check {
    margin-bottom: var(--space-4);
    font-size: var(--t-sm);
  }
  .verdict {
    display: flex;
    gap: var(--space-3);
    align-items: baseline;
  }
  .good {
    color: var(--good);
  }
  .warn {
    color: var(--warn);
  }
  .bad {
    color: var(--bad);
  }
  .plan {
    list-style: none;
    margin: var(--space-2) 0 0;
    padding: 0;
    font-family: var(--font-mono);
    font-size: var(--t-xs);
    color: var(--muted);
    display: flex;
    flex-wrap: wrap;
    gap: 0.2rem var(--space-3);
  }
  .plan .warn {
    color: var(--warn);
  }
  .findings {
    list-style: none;
    margin: var(--space-2) 0 0;
    padding: 0;
    overflow-wrap: anywhere;
  }
</style>
