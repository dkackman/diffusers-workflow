<script lang="ts">
  import type { StepProgress } from '../progress'
  import { formatApprox, formatDuration } from '../format'

  // The Progress panel: one row per step the run will take, the running
  // one carrying its denoise bar, ETA and phase. Once the run is over the
  // panel is one line - how many steps finished and the time it took - with
  // the list behind it
  let {
    steps,
    finishedSteps,
    listStep,
    running,
    progress,
    etaSeconds,
    stall = null,
    took = null,
  }: {
    steps: string[]
    finishedSteps: string[]
    listStep: string | undefined
    running: boolean
    progress: StepProgress
    etaSeconds: number | null
    stall?: { phase: string; seconds: number } | null
    /** Seconds the finished run took, when it says */
    took?: number | null
  } = $props()

  const denoise = $derived(progress.denoise)
  const done = $derived(steps.filter((s) => finishedSteps.includes(s)).length)
</script>

{#if running}
  <h2>Progress</h2>
  {@render list()}
{:else}
  <details class="progress">
    <summary>
      <h2>Progress</h2>
      <span class="muted">
        {done} of {steps.length} steps{took === null
          ? ''
          : ` · ${formatDuration(took)}`}
      </span>
    </summary>
    {@render list()}
  </details>
{/if}

{#snippet list()}
  {#each steps as step (step)}
    <div class="step">
      <span
        class="dot"
        class:done={finishedSteps.includes(step)}
        class:active={step === listStep && running}
      ></span>
      <span class:muted={step !== listStep && !finishedSteps.includes(step)}>
        {step}
      </span>
      {#if step === listStep && running}
        {#if denoise}
          <div class="bar">
            <div
              class="fill"
              style:width={denoise.total_steps
                ? (100 * denoise.step) / denoise.total_steps + '%'
                : '100%'}
            ></div>
          </div>
          <span class="muted count">
            {denoise.step}{denoise.total_steps
              ? ` / ${denoise.total_steps}`
              : ''}
            {#if etaSeconds !== null}· {formatApprox(etaSeconds)} left in this step{/if}
          </span>
        {/if}
        <!-- The counter tells the generating story on its own; every
           other phase is time the bar cannot account for -->
        {#if progress.label && (!denoise || progress.phase !== 'generating')}
          <span class="muted phase" title="what this step is doing now">
            {progress.label}
          </span>
        {/if}
        {#if stall}
          <span
            class="stall"
            title="the engine reports no progress event - long silences are normal for some models"
          >
            no event for {formatDuration(stall.seconds)} in {stall.phase}
          </span>
        {/if}
      {/if}
    </div>
  {/each}
{/snippet}

<style>
  /* One line: the heading inline beside the count, the marker kept */
  .progress summary {
    cursor: pointer;
  }
  .progress summary h2 {
    display: inline;
    margin: 0 var(--space-3) 0 0;
  }
  .progress[open] summary {
    margin-bottom: var(--space-2);
  }
  .stall {
    font-size: 0.82rem;
    color: var(--warn);
  }
  .phase {
    font-size: 0.82rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .step {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.3rem 0;
  }
  .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--panel-2);
    border: 1px solid var(--line);
  }
  .dot.done {
    background: var(--good);
    border-color: var(--good);
  }
  /* The step the worker is on right now - machine state, so it takes the
     signal colour rather than the interactive ink */
  .dot.active {
    background: var(--live);
    border-color: var(--live);
    animation: dw-pulse 1.6s ease-in-out infinite;
  }
  @media (prefers-reduced-motion: reduce) {
    .dot.active {
      animation: none;
    }
  }
  .bar {
    flex: 1;
    max-width: 340px;
    height: 8px;
    border-radius: 4px;
    background: var(--panel-2);
    overflow: hidden;
  }
  .fill {
    height: 100%;
    background: var(--live);
    transition: width 0.3s;
  }
  .count {
    font-variant-numeric: tabular-nums;
    font-size: 0.8rem;
  }
</style>
