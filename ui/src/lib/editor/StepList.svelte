<script lang="ts">
  import { Plus } from '@lucide/svelte'
  import StepEditor from './StepEditor.svelte'
  import { emptyStep, emptyTaskStep, emptyWorkflowStep } from '../editor'
  import type { StepFlow } from '../flow'
  import type { StepModes } from '../stepModes.svelte'

  // The workflow's steps down a numbered rail, with the controls that add
  // them and set how much of each shows
  let {
    steps = $bindable(),
    modes,
    folder,
    flow,
    problemsByStep,
    stepReferences,
    hovered = $bindable(),
  }: {
    steps: Record<string, any>[] | undefined
    modes: StepModes
    // The folder the workflow saves into, which a sub-workflow step's
    // relative path resolves against
    folder: string
    flow: StepFlow[]
    problemsByStep: Map<number, string[]>
    stepReferences: string[][]
    hovered: string | null
  } = $props()

  function addStep(kind: string) {
    const step =
      kind === 'task'
        ? emptyTaskStep()
        : kind === 'workflow'
          ? emptyWorkflowStep()
          : emptyStep()
    // Step mode and flow-graph edges are keyed by name - two steps sharing
    // the factory default (e.g. 'generate') would collide in the step modes
    const existingNames = new Set(
      (steps ?? []).map((s: Record<string, any>) => s.name),
    )
    if (existingNames.has(step.name)) {
      let n = 2
      while (existingNames.has(`${step.name}-${n}`)) n++
      step.name = `${step.name}-${n}`
    }
    steps = [...(steps ?? []), step]
    modes.mark(step.name, 'full')
  }

  function removeStep(index: number) {
    steps!.splice(index, 1)
  }

  function moveStep(index: number, delta: number) {
    const list = steps!
    const target = index + delta
    if (target < 0 || target >= list.length) return
    ;[list[index], list[target]] = [list[target], list[index]]
  }
</script>

{#if (steps ?? []).length > 1}
  <div class="densityrow">
    <span class="muted">steps</span>
    <span class="flex"></span>
    <button class="quiet" onclick={() => modes.setAll(steps ?? [], 'collapsed')}
      >collapse all</button
    >
    <button class="quiet" onclick={() => modes.setAll(steps ?? [], 'compact')}
      >compact all</button
    >
    <button class="quiet" onclick={() => modes.setAll(steps ?? [], 'full')}
      >expand all</button
    >
  </div>
{/if}

<div class="steps">
  {#each steps ?? [] as step, index (step)}
    <div
      id={'step-' + step.name}
      class="steprow"
      class:flowlit={hovered !== null && step.name === hovered}
    >
      <div class="railcell">
        <span class="ordinal" title={`step ${index + 1} of ${steps!.length}`}
          >{index + 1}</span
        >
      </div>
      <StepEditor
        bind:step={steps![index]}
        {index}
        count={steps!.length}
        references={stepReferences[index] ?? []}
        baseFolder={folder}
        mode={modes.of(step)}
        flow={flow[index]}
        problems={problemsByStep.get(index) ?? []}
        onmodechange={(m) => modes.set(step, m)}
        onhover={(n) => (hovered = n)}
        onremove={() => removeStep(index)}
        onmove={(delta) => moveStep(index, delta)}
      />
    </div>
  {/each}
</div>

<div class="addstep">
  <button
    class="quiet withicon"
    onclick={() => addStep('pipeline')}
    title="add a step that runs a diffusers pipeline"
  >
    <Plus size={14} />pipeline step
  </button>
  <button
    class="quiet withicon"
    onclick={() => addStep('task')}
    title="add a utility step - upscaling, segmentation, captioning, frame tools"
  >
    <Plus size={14} />task step
  </button>
  <button
    class="quiet withicon"
    onclick={() => addStep('workflow')}
    title="add a step that runs another workflow file with mapped arguments"
  >
    <Plus size={14} />sub-workflow step
  </button>
</div>

<style>
  .addstep {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem 0.6rem;
  }
  .densityrow {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: var(--space-2);
  }
  .densityrow button {
    font-size: 0.75rem;
    padding: 0.2rem 0.55rem;
  }
  .steps {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    margin-bottom: var(--space-4);
  }
  .steprow {
    display: grid;
    grid-template-columns: 28px minmax(0, 1fr);
    gap: 0 var(--space-2);
  }
  .railcell {
    position: relative;
    display: flex;
    justify-content: center;
  }
  /* the connecting line - drawn per row so it spans the gaps too */
  .steprow:not(:last-child) .railcell::before {
    content: '';
    position: absolute;
    top: 26px;
    bottom: calc(-1 * var(--space-3));
    width: 2px;
    background: color-mix(in srgb, var(--accent) 35%, transparent);
  }
  .ordinal {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: var(--accent);
    color: var(--accent-ink);
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-top: var(--space-2);
    z-index: 1;
  }
  .steprow.flowlit :global(.panel.step) {
    border-color: var(--accent);
  }
</style>
