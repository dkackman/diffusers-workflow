<script lang="ts">
  import type { WorkflowDefinition } from '../types'
  import { CircleCheck, TriangleAlert, X } from '@lucide/svelte'
  import { describePlan } from '../plan'
  import { placeFindings } from '../findings'
  import { findingPlace, type Place } from '../findingPlace'
  import { verdictOf } from '../runCheck.svelte'
  import type { ValidationResult } from '../types'
  import PlanLines from '../workflow/PlanLines.svelte'
  import ErrorLine from '../ui/ErrorLine.svelte'

  // The verdict on the definition as it stands, and under it, whenever the
  // server sent a plan, what a run will do. The reader is the author, so
  // every warning stays visible: nothing is split off for one. An error
  // that names a step or variable `doc` still has offers to go to it
  // A check that failed outright has no answer: `error` says so instead
  let {
    validation,
    error = null,
    doc,
    ondismiss,
    onjump,
  }: {
    validation: ValidationResult | null
    error?: unknown
    doc: WorkflowDefinition
    ondismiss: () => void
    onjump: (place: Place) => void
  } = $props()

  const placed = $derived(
    validation
      ? placeFindings(validation, [])
      : { general: [], author: [], byVariable: {} },
  )
  const verdict = $derived(verdictOf(validation, error, placed))
  // A plan line that warns is the thing to read: its details open
  const planWarns = $derived(
    validation?.plan
      ? describePlan(validation.plan).some((line) => line.tone === 'warn')
      : false,
  )
</script>

<div
  class="panel validation"
  class:error-edge={!validation || !validation.valid}
  class:warn-edge={validation?.valid && validation.warnings.length > 0}
  class:good-edge={validation?.valid && validation.warnings.length === 0}
>
  <button
    class="quiet icon dismiss"
    onclick={ondismiss}
    title="dismiss"
    aria-label="dismiss validation results"
  >
    <X size={13} />
  </button>
  {#if verdict}
    <div class="verdict {verdict.tone}">
      {#if verdict.tone === 'good'}<CircleCheck size={14} />{/if}
      {verdict.text}
      {#if verdict.remedy}<span class="muted">{verdict.remedy}</span>{/if}
    </div>
  {/if}
  {#if error !== null && error !== undefined}<ErrorLine {error} />{/if}
  {#each placed.general as finding, i (i)}
    {@const place = findingPlace(finding.path ?? null, doc)}
    <div class="error">
      {finding.text}
      {#if place}
        <button class="quiet goto" onclick={() => onjump(place)}>
          Go to {place.kind}
          {place.name}
        </button>
      {/if}
    </div>
  {/each}
  {#each placed.author as finding, i (i)}
    <div class="warn"><TriangleAlert size={14} /> {finding.text}</div>
  {/each}
  {#if validation?.plan}
    <!-- What a run of this definition, with these defaults, will do -
         the figure to expect before pressing Run, and the weights it
         would pull first, which the cost block never counts. Shown whenever
         the server sent one: a repo that isn't there makes the answer
         invalid but keeps its plan, under the error that names it -->
    <PlanLines plan={validation.plan} open={planWarns} />
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
  .verdict {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
  }
  .verdict.good {
    color: var(--good);
  }
  .goto {
    margin-left: 0.4rem;
    padding: 0.05rem 0.45rem;
    font-size: 0.8rem;
  }
  .warn {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  /* A finding leads with an engine path that has no spaces to break at */
  .error,
  .warn {
    overflow-wrap: anywhere;
  }
</style>
