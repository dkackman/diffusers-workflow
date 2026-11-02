<script lang="ts">
  import { dataFlowGraph, type FlowNode } from '../flow'
  import { formatDuration } from '../format'
  import {
    BOX_H,
    BOX_W,
    CHIP_CHARS,
    CHIP_CHARS_TIMED,
    CHIP_H,
    CHIP_STEP,
    CHIP_TOP,
    DETAIL_CHARS,
    NAME_CHARS,
    NAME_CHARS_WITH_TAG,
    fit,
    heightOf,
    layoutGraph,
  } from './flowLayout'

  let {
    workflow,
    onselect = undefined,
    selectable = undefined,
    selectHint = 'click a step to open it in the form',
    activeStep = undefined,
    failedStep = undefined,
    doneSteps = [],
    activeMember = undefined,
    failedMember = undefined,
    doneMembers = [],
    nodeTimes = {},
    memberTimes = {},
  }: {
    workflow: Record<string, any>
    onselect?: (stepName: string) => void
    /** Which steps onselect has somewhere to take the reader - every step
     * when not given. A step it rules out is drawn as no control at all. */
    selectable?: (stepName: string) => boolean
    /** What a click on a step does, for the legend. */
    selectHint?: string
    /** The step a run is executing right now, when the graph is showing a
     * live job rather than a workflow being edited. */
    activeStep?: string
    /** The step a failed run stopped in. */
    failedStep?: string
    /** Steps that run has already finished. */
    doneSteps?: string[]
    /** The for_each member (`group@entry`) the run is on right now - a
     * finer grain than activeStep, which names the group. */
    activeMember?: string
    /** The for_each member a failed run stopped in. */
    failedMember?: string
    /** The for_each members that run has already finished, engine names. */
    doneMembers?: string[]
    /** Seconds each finished node took, from the run's own event clocks. */
    nodeTimes?: Record<string, number>
    /** Seconds each finished for_each member took, engine names. */
    memberTimes?: Record<string, number>
  } = $props()

  const showsRun = $derived(
    activeStep !== undefined ||
      failedStep !== undefined ||
      doneSteps.length > 0,
  )
  // On a run, a step that is none of running, failed or finished has not
  // run yet - drawn faded, so what the run got through reads at a glance
  const stateOf = $derived((name: string) =>
    name === activeStep
      ? 'active'
      : name === failedStep
        ? 'failed'
        : doneSteps.includes(name)
          ? 'done'
          : showsRun
            ? 'pending'
            : '',
  )
  const memberStateOf = $derived((full: string) =>
    full === activeMember
      ? 'active'
      : full === failedMember
        ? 'failed'
        : doneMembers.includes(full)
          ? 'done'
          : '',
  )

  const graph = $derived(dataFlowGraph(workflow))
  const hasForEach = $derived(graph.nodes.some((node) => node.forEach))
  const hasPending = $derived(
    graph.nodes.some((node) => stateOf(node.name) === 'pending'),
  )
  // One plain step has no flow to draw: it is shown as a line, not a canvas
  const lone = $derived(
    graph.nodes.length === 1 && !graph.nodes[0].forEach
      ? graph.nodes[0]
      : undefined,
  )
  const canSelect = (name: string) => !!onselect && (selectable?.(name) ?? true)

  /** The parts of a node's labels that did not fit, in full, for its tooltip. */
  function overflowTitle(node: FlowNode): string {
    const nameShown = fit(
      node.name,
      node.isEntryPoint ? NAME_CHARS_WITH_TAG : NAME_CHARS,
      'head',
    )
    return [
      nameShown === node.name ? null : node.name,
      fit(node.detail, DETAIL_CHARS, 'tail') === node.detail
        ? null
        : node.detail,
    ]
      .filter(Boolean)
      .join('\n')
  }

  // The geometry - box positions, lanes, edge paths - is flowLayout.ts;
  // the template draws what it returns
  const layout = $derived(layoutGraph(graph))

  function kindLabel(kind: string): string {
    if (kind === 'pipeline') return 'pipeline'
    if (kind === 'task') return 'task'
    if (kind === 'workflow') return 'sub-workflow'
    return 'step'
  }

  function select(name: string) {
    onselect?.(name)
  }
  function onKeydown(event: KeyboardEvent, name: string) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      select(name)
    }
  }

  // Without a consumer for the click - the job page just shows the shape of
  // the run - a node is not a control: no focus stop, no pointer, nothing
  // announced as a button.
  const nodeAttributes = (name: string) =>
    canSelect(name)
      ? {
          role: 'button',
          tabindex: 0,
          onclick: () => select(name),
          onkeydown: (event: KeyboardEvent) => onKeydown(event, name),
        }
      : {}
</script>

{#if (graph.nodes ?? []).length === 0}
  <p class="muted">No steps yet.</p>
{:else if lone}
  {@const state = stateOf(lone.name)}
  <section class="flowwrap panel" aria-label="workflow data-flow">
    <div
      class="loneline {state}"
      class:clickable={canSelect(lone.name)}
      aria-label={`step ${lone.name}, ${kindLabel(lone.kind)}${state ? ', ' + state : ''}`}
      {...nodeAttributes(lone.name)}
    >
      <span class="dot"></span>
      <strong>{lone.name}</strong>
      <span class="lonekind">{kindLabel(lone.kind)}</span>
      {#if lone.detail}<code class="muted">{lone.detail}</code>{/if}
      <span class="flex"></span>
      {#if state === 'done' && nodeTimes[lone.name] !== undefined}
        <code class="muted">{formatDuration(nodeTimes[lone.name])}</code>
      {:else if state === 'active' || state === 'failed'}
        <span class="muted">{state === 'active' ? 'running' : 'failed'}</span>
      {/if}
    </div>
  </section>
{:else}
  <section class="flowwrap panel" aria-label="workflow data-flow">
    <!-- A key to the marks this graph actually uses, each worded in full on
         hover; nothing listed for a mark the diagram does not draw -->
    <ul class="legend muted" aria-label="data-flow legend">
      <li title="a step with no previous_result input - it starts the run">
        <span class="swatch entry"></span>entry
      </li>
      {#if graph.edges.length}
        <li
          title="a previous_result reference, labeled with the argument it feeds"
        >
          <span class="swatch arrow"></span><code>previous_result</code>
        </li>
      {/if}
      {#if graph.fanIn.size}
        <li
          title="more than one incoming arrow: the step's inputs multiply together (docs/WORKFLOW_GUIDE.md, 'Several previous_result references multiply')"
        >
          <span class="fanmark">×</span>inputs multiply
        </li>
      {/if}
      {#if hasForEach}
        <li title="a for_each step: the entries it runs, inset">
          <span class="swatch chip"></span><code>for_each</code> entries
        </li>
      {/if}
      {#if doneSteps.length}
        <li title="this run has finished the step; its time is on the right">
          <span class="swatch done"></span>finished
        </li>
      {/if}
      {#if activeStep !== undefined}
        <li title="the step this run is executing now">
          <span class="swatch active"></span>running
        </li>
      {/if}
      {#if failedStep !== undefined}
        <li title="the step this run failed in - the error is below">
          <span class="swatch failed"></span>failed
        </li>
      {/if}
      {#if hasPending}
        <li title="a step this run has not reached">
          <span class="swatch pending"></span>not run
        </li>
      {/if}
      {#if onselect}
        <li class="tip">{selectHint}</li>
      {/if}
    </ul>
    <div class="scrollarea">
      <svg
        width={layout.width}
        height={layout.height}
        viewBox={`0 0 ${layout.width} ${layout.height}`}
        role="img"
        aria-label="workflow data-flow diagram"
      >
        <defs>
          <marker
            id="flow-arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" class="arrowhead" />
          </marker>
          <clipPath id="flow-nodebox">
            <rect width={BOX_W} height={BOX_H} rx="8" />
          </clipPath>
          {#each layout.tallHeights as h (h)}
            <clipPath id={`flow-nodebox-${h}`}>
              <rect width={BOX_W} height={h} rx="8" />
            </clipPath>
          {/each}
        </defs>

        {#each layout.edgeLines ?? [] as edge, i (i)}
          {#if edge}
            <path d={edge.path} class="edge" marker-end="url(#flow-arrow)" />
          {/if}
        {/each}
        <!-- Labels after every line, so a label's halo masks a line that
             crosses it rather than the line striking the label through -->
        {#each layout.edgeLines ?? [] as edge, i (i)}
          {#if edge}
            <text x={edge.labelX} y={edge.labelY} class="edgelabel"
              >{#if edge.label !== edge.title}<title>{edge.title}</title
                >{/if}{edge.label}</text
            >
          {/if}
        {/each}

        {#each graph.nodes as node (node.name)}
          {@const pos = layout.positions[node.name]}
          {#if pos}
            {@const fanIn = graph.fanIn.get(node.name)}
            {@const boxH = heightOf(node)}
            <g
              class="node"
              class:entry={node.isEntryPoint}
              class:fanin={!!fanIn}
              class:clickable={canSelect(node.name)}
              class:active={node.name === activeStep}
              class:failed={stateOf(node.name) === 'failed'}
              class:pending={stateOf(node.name) === 'pending'}
              class:done={stateOf(node.name) === 'done'}
              transform={`translate(${pos.x}, ${pos.y})`}
              clip-path={boxH > BOX_H
                ? `url(#flow-nodebox-${boxH})`
                : 'url(#flow-nodebox)'}
              aria-label={`step ${node.name}, ${kindLabel(node.kind)}${stateOf(node.name) ? ', ' + stateOf(node.name) : ''}${node.isEntryPoint ? ', entry point' : ''}${fanIn ? ', fan-in: ' + fanIn.label : ''}${node.forEach ? (node.members?.length ? `, for_each with ${node.members.length} entries` : ', for_each') : ''}`}
              {...nodeAttributes(node.name)}
            >
              {#if overflowTitle(node)}
                <title>{overflowTitle(node)}</title>
              {/if}
              <rect width={BOX_W} height={boxH} rx="8" class="box" />
              <text x="10" y="20" class="stepname"
                >{fit(
                  node.name,
                  node.isEntryPoint ? NAME_CHARS_WITH_TAG : NAME_CHARS,
                  'head',
                )}</text
              >
              <text x="10" y="37" class="stepkind">{kindLabel(node.kind)}</text>
              {#if stateOf(node.name) === 'done' && nodeTimes[node.name] !== undefined}
                <text x={BOX_W - 8} y="37" class="steptime" text-anchor="end"
                  >{formatDuration(nodeTimes[node.name])}</text
                >
              {/if}
              {#if node.detail}
                <text x="10" y="52" class="stepdetail"
                  >{fit(node.detail, DETAIL_CHARS, 'tail')}</text
                >
              {/if}
              {#if node.isEntryPoint}
                <text x={BOX_W - 8} y="14" class="entrytag" text-anchor="end"
                  >entry</text
                >
              {/if}
              {#if node.members?.length}
                <!-- The run's entries, in the order the engine expands them:
                     each chip named as the workflow wrote it, the engine's
                     own name for it on hover -->
                {#each node.members as key, i (i)}
                  {@const full = `${node.name}@${key}`}
                  {@const state = memberStateOf(full)}
                  {@const took =
                    state === 'done' ? memberTimes[full] : undefined}
                  <g
                    class="member"
                    class:done={state === 'done'}
                    class:active={state === 'active'}
                    class:failed={state === 'failed'}
                    aria-label={`member ${full}${state ? ', ' + state : ''}`}
                  >
                    <title>{full}</title>
                    <rect
                      x="10"
                      y={CHIP_TOP + i * CHIP_STEP}
                      width={BOX_W - 20}
                      height={CHIP_H}
                      rx="4"
                      class="chip"
                    />
                    <text
                      x="15"
                      y={CHIP_TOP + i * CHIP_STEP + 11}
                      class="chiplabel"
                      >{fit(
                        key,
                        took === undefined ? CHIP_CHARS : CHIP_CHARS_TIMED,
                        'head',
                      )}</text
                    >
                    {#if took !== undefined}
                      <text
                        x={BOX_W - 15}
                        y={CHIP_TOP + i * CHIP_STEP + 11}
                        class="steptime"
                        text-anchor="end">{formatDuration(took)}</text
                      >
                    {/if}
                  </g>
                {/each}
              {:else if node.forEach}
                <text x="11" y={CHIP_TOP + 12} class="membersunknown"
                  >for_each</text
                >
              {/if}
            </g>
            {#if fanIn}
              <text
                x={pos.x + BOX_W / 2}
                y={pos.y + boxH + 14}
                class="fanlabel"
                text-anchor="middle">× {fanIn.label}</text
              >
            {/if}
          {/if}
        {/each}
      </svg>
    </div>
  </section>
{/if}

<style>
  .flowwrap {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
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
  /* A lone step: one line in place of the canvas, its run state on the
     dot as the diagram puts it on the border */
  .loneline {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: var(--space-1) var(--space-3);
    padding: var(--space-2) var(--space-3);
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--panel-2);
  }
  .loneline.clickable {
    cursor: pointer;
  }
  .loneline.clickable:hover,
  .loneline.clickable:focus-visible {
    border-color: var(--accent);
  }
  .loneline code {
    font-family: var(--font-mono);
    font-size: 0.78rem;
  }
  .lonekind {
    font-size: 0.8rem;
    color: var(--accent);
  }
  .loneline .flex {
    flex: 1;
  }
  .dot {
    align-self: center;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    border: 1.5px solid var(--line);
  }
  .loneline.done .dot {
    background: var(--good);
    border-color: var(--good);
  }
  .loneline.failed .dot {
    background: var(--bad);
    border-color: var(--bad);
  }
  .loneline.active .dot {
    background: var(--live);
    border-color: var(--live);
    animation: dw-pulse 1.6s ease-in-out infinite;
  }
  @media (prefers-reduced-motion: reduce) {
    .loneline.active .dot {
      animation: none;
    }
  }

  /* As wide as the graph, no wider: a two-step workflow is not a
     panel-wide grey field with its boxes in one corner */
  .scrollarea {
    align-self: flex-start;
    max-width: 100%;
    overflow: auto;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--panel-2);
  }
  svg {
    display: block;
  }
  .edge {
    fill: none;
    stroke: var(--muted);
    stroke-width: 1.5;
    opacity: 0.75;
  }
  .arrowhead {
    fill: var(--muted);
  }
  .edgelabel {
    font-size: 10px;
    fill: var(--muted);
    text-anchor: middle;
    font-family: var(--font-mono);
    paint-order: stroke;
    stroke: var(--panel-2);
    stroke-width: 3px;
    stroke-linejoin: round;
  }
  .node.clickable {
    cursor: pointer;
  }
  .box {
    fill: var(--panel);
    stroke: var(--line);
    stroke-width: 1.5;
  }
  .node.clickable:hover .box,
  .node.clickable:focus-visible .box {
    stroke: var(--accent);
  }
  .node.entry .box {
    stroke-dasharray: 4 3;
  }
  /* Fan-in is marked by its × label alone: the border belongs to run
     state, and --warn is the same amber as --live, so a fan-in border read
     as "running" */
  .node.pending {
    opacity: 0.45;
  }
  .node.done .box {
    stroke: var(--good);
    stroke-width: 2;
  }
  .node.failed .box {
    stroke: var(--bad);
    stroke-width: 2.5;
  }
  /* The step running right now - machine state, so the signal colour */
  .node.active .box {
    stroke: var(--live);
    stroke-width: 2.5;
    animation: dw-pulse 1.6s ease-in-out infinite;
  }
  @media (prefers-reduced-motion: reduce) {
    .node.active .box {
      animation: none;
    }
  }
  .stepname {
    font-size: 12px;
    font-weight: 700;
    fill: var(--ink);
  }
  .stepkind {
    font-size: 10px;
    fill: var(--accent);
    text-transform: none;
  }
  .stepdetail {
    font-size: 10px;
    fill: var(--muted);
    font-family: var(--font-mono);
  }
  /* a finished step's run time, read off the event clocks */
  .steptime {
    font-size: 10px;
    fill: var(--muted);
    font-family: var(--font-mono);
  }
  /* muted, not green: green is "finished" here, and the dashed border
     already says entry */
  .entrytag {
    font-size: 9px;
    fill: var(--muted);
    text-transform: none;
  }
  .fanlabel {
    font-size: 10px;
    fill: var(--ink);
    font-weight: 700;
  }
  /* The entries a for_each step runs, inset beneath its header. Machine
     state on their edges, as on the nodes: done green, running the
     safelight amber. */
  .member .chip {
    fill: var(--panel-2);
    stroke: var(--line);
    stroke-width: 1;
  }
  .member.done .chip {
    stroke: var(--good);
    stroke-width: 1.5;
  }
  .member.failed .chip {
    stroke: var(--bad);
    stroke-width: 1.5;
  }
  .member.active .chip {
    stroke: var(--live);
    stroke-width: 1.5;
    animation: dw-pulse 1.6s ease-in-out infinite;
  }
  @media (prefers-reduced-motion: reduce) {
    .member.active .chip {
      animation: none;
    }
  }
  .chiplabel {
    font-size: 9px;
    fill: var(--ink);
    font-family: var(--font-mono);
  }
  .membersunknown {
    font-size: 9px;
    fill: var(--muted);
    font-family: var(--font-mono);
  }
</style>
