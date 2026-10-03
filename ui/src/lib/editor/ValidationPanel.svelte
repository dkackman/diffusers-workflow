<script lang="ts">
  import { CircleCheck, TriangleAlert, X } from '@lucide/svelte'
  import { describePlan } from '../plan'
  import type { ValidationResult } from '../types'

  // The verdict on the definition as it stands, and under a valid one what
  // a run will do
  let {
    validation,
    ondismiss,
  }: { validation: ValidationResult; ondismiss: () => void } = $props()
</script>

<div
  class="panel validation"
  class:error-edge={!validation.valid}
  class:warn-edge={validation.valid && validation.warnings.length > 0}
  class:good-edge={validation.valid && validation.warnings.length === 0}
>
  <button
    class="quiet icon dismiss"
    onclick={ondismiss}
    title="dismiss"
    aria-label="dismiss validation results"
  >
    <X size={13} />
  </button>
  {#if validation.valid && !validation.warnings.length}
    <span class="ok"
      ><CircleCheck size={14} /> schema-valid, no argument warnings</span
    >
  {:else if validation.valid}
    {#each validation.warnings as warning, i (i)}
      <div class="warn"><TriangleAlert size={14} /> {warning}</div>
    {/each}
  {:else if validation.errors && validation.errors.length}
    {#each validation.errors as e, i (i)}
      <div class="error">{e.path ?? 'root'}: {e.message}</div>
    {/each}
  {:else}
    <div class="error">{validation.error}</div>
  {/if}
  {#if validation.valid && validation.plan}
    <!-- What a run of this definition, with these defaults, will do -
         the figure to expect before pressing Run, and the weights it
         would pull first, which the cost block never counts -->
    <ul
      class="plan"
      aria-label="what a run will do"
      title="from the workflow's own cost block and this server's model cache - a plan, not a promise"
    >
      {#each describePlan(validation.plan) as line, i (i)}
        <li class:warn={line.tone === 'warn'}>{line.text}</li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .panel {
    margin-bottom: 1rem;
  }
  .validation {
    position: relative;
    padding-right: 2.2rem;
  }
  .dismiss {
    position: absolute;
    top: var(--space-2);
    right: var(--space-2);
    border: 0;
    padding: 0.2rem 0.3rem;
  }
  .ok {
    color: var(--good);
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
  }
  .warn {
    color: var(--warn);
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .error {
    color: var(--bad);
  }
  /* The plan under the verdict: figures the engine resolves, so mono, and
     quiet - a warn line is the one that changes the number to expect */
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
    display: inline;
  }
</style>
