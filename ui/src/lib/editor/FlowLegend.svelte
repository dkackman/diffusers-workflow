<script lang="ts">
  // FlowView's key: one entry for each mark the diagram actually draws,
  // each worded in full on hover; nothing listed for a mark it does not draw
  let {
    edges = false,
    fanIn = false,
    forEach = false,
    chain = false,
    done = false,
    active = false,
    failed = false,
    pending = false,
    hint = undefined,
  }: {
    edges?: boolean
    fanIn?: boolean
    forEach?: boolean
    chain?: boolean
    done?: boolean
    active?: boolean
    failed?: boolean
    pending?: boolean
    /** What a click on a step does; absent when a step is no control. */
    hint?: string
  } = $props()
</script>

<ul class="legend muted" aria-label="data-flow legend">
  <li title="a step with no previous_result input - it starts the run">
    <span class="swatch entry"></span>entry
  </li>
  {#if edges}
    <li title="a previous_result reference, labeled with the argument it feeds">
      <span class="swatch arrow"></span><code>previous_result</code>
    </li>
  {/if}
  {#if fanIn}
    <li
      title="more than one incoming arrow: the step's inputs multiply together (docs/WORKFLOW_GUIDE.md, 'Several previous_result references multiply')"
    >
      <span class="fanmark">×</span>inputs multiply
    </li>
  {/if}
  {#if forEach}
    <li title="a for_each step: the entries it runs, inset">
      <span class="swatch chip"></span><code>for_each</code> entries
    </li>
  {/if}
  {#if chain}
    <li
      title="a chained step: the pipeline runs once per segment, inset, each seam's carry between them, and the segments are stitched into one video"
    >
      <span class="swatch chip"></span><code>chain</code> segments
    </li>
  {/if}
  {#if done}
    <li title="this run has finished the step; its time is on the right">
      <span class="swatch done"></span>finished
    </li>
  {/if}
  {#if active}
    <li title="the step this run is executing now">
      <span class="swatch active"></span>running
    </li>
  {/if}
  {#if failed}
    <li title="the step this run failed in - the error is below">
      <span class="swatch failed"></span>failed
    </li>
  {/if}
  {#if pending}
    <li title="a step this run has not reached">
      <span class="swatch pending"></span>not run
    </li>
  {/if}
  {#if hint}
    <li class="tip">{hint}</li>
  {/if}
</ul>

<style>
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-3);
    margin: 0;
    padding: 0;
    list-style: none;
    font-size: 0.8rem;
  }
  .legend li {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    cursor: help;
  }
  .legend li.tip {
    cursor: default;
    margin-left: auto;
  }
  .legend code {
    font-family: var(--font-mono);
    font-size: 0.75rem;
  }
  /* Each swatch is a miniature of the mark it names, stroked as the
     diagram strokes it */
  .swatch {
    display: inline-block;
    width: 18px;
    height: 12px;
    border-radius: 3px;
    border: 1.5px solid var(--line);
    background: var(--panel);
  }
  .swatch.entry {
    border-style: dashed;
    border-color: var(--muted);
  }
  .swatch.arrow {
    border: none;
    background: none;
    height: 2px;
    border-top: 1.5px solid var(--muted);
    position: relative;
  }
  .swatch.arrow::after {
    content: '';
    position: absolute;
    right: -1px;
    top: -5px;
    border: 4px solid transparent;
    border-left: 6px solid var(--muted);
  }
  .fanmark {
    font-weight: 700;
    color: var(--ink);
  }
  .swatch.chip {
    height: 8px;
    border-width: 1px;
    background: var(--panel-2);
  }
  .swatch.done {
    border: 2px solid var(--good);
  }
  .swatch.active {
    border: 2.5px solid var(--live);
  }
  .swatch.failed {
    border: 2.5px solid var(--bad);
  }
  .swatch.pending {
    opacity: 0.45;
  }
</style>
