<script lang="ts">
  // What validate says a Run with these values would do, checked as you
  // edit: the verdict in plain words, the plan, what stands in the way, and
  // behind a fold what only the workflow's author can act on. Every finding
  // is server text, shown as text
  import type { Finding } from '../findings'
  import ErrorLine from '../ui/ErrorLine.svelte'
  import Details from '../ui/Details.svelte'
  import type { Plan } from '../types'
  import type { Verdict } from '../runCheck.svelte'
  import PlanLines from '../ui/PlanLines.svelte'

  let {
    verdict,
    checking,
    plan,
    checkError,
    findings,
    author,
  }: {
    verdict: Verdict | null
    checking: boolean
    plan: Plan | null
    /** A check the server refused outright, as caught, first in the list */
    checkError: unknown
    /** What validate said that names no field the form shows */
    findings: Finding[]
    /** Warnings no field the form shows is the place for */
    author: Finding[]
  } = $props()
</script>

<section class="panel check" aria-label="run check" aria-busy={checking}>
  <div class="verdict">
    {#if verdict}
      <span class={verdict.tone}>{verdict.text}</span>
      {#if verdict.remedy}<span class="muted">{verdict.remedy}</span>{/if}
    {/if}
    {#if checking}<span class="muted">Checking…</span>{/if}
  </div>
  {#if plan}<PlanLines {plan} />{/if}
  {#if checkError != null || findings.length}
    <ul class="findings" aria-label="findings">
      {#if checkError != null}
        <li class="bad">
          <ErrorLine error={checkError} doing="check these values" />
        </li>
      {/if}
      {#each findings as finding, i (i)}
        <li id="issue-{i}" class={finding.tone}>{finding.text}</li>
      {/each}
    </ul>
  {/if}
  {#if author.length}
    <Details
      summary="{author.length} {author.length === 1
        ? 'note'
        : 'notes'} for the workflow's author"
      detail={author.map((f) => f.text).join('\n')}
    />
  {/if}
</section>

<style>
  /* The check beside Run: the verdict, then the plan, then what validate
     said that names no field */
  .check {
    margin-bottom: var(--space-4);
    font-size: var(--t-sm);
  }
  .verdict {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-3);
    align-items: baseline;
  }
  .findings {
    list-style: none;
    margin: var(--space-2) 0 0;
    padding: 0;
    overflow-wrap: anywhere;
  }
</style>
