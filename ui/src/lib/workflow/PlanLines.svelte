<script lang="ts">
  // What a job would do, as one line to read and the lines behind it: the
  // run check and the editor's validation both show a plan this way
  import { describePlan, planSummary } from '../plan'
  import type { Plan } from '../types'
  import Details from '../ui/Details.svelte'

  let { plan, open = false }: { plan: Plan; open?: boolean } = $props()
</script>

<div
  class="plan"
  title="from the workflow's own cost block and this server's model cache - a plan, not a promise"
>
  <p class="num">{planSummary(plan)}</p>
  <Details
    summary="What a job will do"
    detail={describePlan(plan)
      .map((line) => line.text)
      .join('\n')}
    {open}
  />
</div>

<style>
  .plan {
    margin-top: var(--space-2);
    color: var(--muted);
  }
  .plan p {
    margin: 0;
    font-family: var(--font-mono);
    font-size: var(--t-xs);
  }
</style>
