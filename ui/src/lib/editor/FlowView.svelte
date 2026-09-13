<script lang="ts">
  import { dataFlowGraph, type FlowNode } from '../flow'

  let {
    workflow,
    onselect = undefined,
    activeStep = undefined,
    doneSteps = [],
    activeMember = undefined,
    doneMembers = [],
  }: {
    workflow: Record<string, any>
    onselect?: (stepName: string) => void
    /** The step a run is executing right now, when the graph is showing a
     * live job rather than a workflow being edited. */
    activeStep?: string
    /** Steps that run has already finished. */
    doneSteps?: string[]
    /** The for_each member (`group@entry`) the run is on right now - a
     * finer grain than activeStep, which names the group. */
    activeMember?: string
    /** The for_each members that run has already finished, engine names. */
    doneMembers?: string[]
  } = $props()

  const showsRun = $derived(activeStep !== undefined || doneSteps.length > 0)
  const stateOf = $derived((name: string) =>
    name === activeStep ? 'active' : doneSteps.includes(name) ? 'done' : '',
  )
  const memberStateOf = $derived((full: string) =>
    full === activeMember ? 'active' : doneMembers.includes(full) ? 'done' : '',
  )

  const graph = $derived(dataFlowGraph(workflow))

  // SVG text neither wraps nor takes text-overflow, so a label longer than
  // the box ran out of its right edge. Budgets are characters at the box's
  // inner width (BOX_W less the 10px inset each side) for each line's font
  // - bold 12px sans for the name, 10px mono for the detail - and the
  // clipPath below catches what a wider glyph set still pushes past.
  const NAME_CHARS = 20
  const NAME_CHARS_WITH_TAG = 15 // the entry tag sits in the top-right corner
  const DETAIL_CHARS = 26

  /** The text cut to `max` characters with an ellipsis where it was cut: a
   * name is told apart by how it starts, a path by how it ends. */
  function fit(text: string, max: number, keep: 'head' | 'tail'): string {
    if (text.length <= max) return text
    return keep === 'head'
      ? text.slice(0, max - 1) + '…'
      : '…' + text.slice(text.length - max + 1)
  }
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

  // Layered left-to-right layout: a node's layer is one past the deepest
  // producer that feeds it directly, so entry points (no previous_result
  // input) sit in the first column and depth reads as real dependency
  // distance rather than just JSON step order.
  const BOX_W = 176
  const BOX_H = 60
  const COL_W = 232
  const ROW_H = 92
  const PAD = 28
  // Member chips: a list-driven step's box grows to hold one inset chip
  // per entry, beneath the header the ordinary box's three lines occupy
  const CHIP_H = 15
  const CHIP_STEP = 19
  const CHIP_CHARS = 26
  const CHIP_TOP = BOX_H - 2
  // The gap ROW_H left between fixed-height boxes, kept for the
  // height-aware stacking below
  const ROW_GAP = ROW_H - BOX_H

  /** The box's height: the standard header, plus a chip row per entry for
   * a list-driven step - or one empty slot when the step carries
   * for_each but its list cannot be read from the definition. */
  function heightOf(node: FlowNode): number {
    const count = node.members?.length ?? 0
    if (count) return CHIP_TOP + (count - 1) * CHIP_STEP + CHIP_H + 6
    return node.forEach ? BOX_H + 26 : BOX_H
  }

  const layout = $derived.by(() => {
    const { nodes, edges } = graph
    // Plain objects rather than Map/Set here: these are throw-away
    // scratch structures rebuilt on every recompute, not reactive state,
    // and svelte's lint rule wants SvelteMap for anything mutated after
    // construction.
    const producersOf: Record<string, string[]> = {}
    for (const e of edges) {
      producersOf[e.to] = [...(producersOf[e.to] ?? []), e.from]
    }
    const layerOf: Record<string, number> = {}
    function layerFor(name: string, guard: Record<string, true>): number {
      if (name in layerOf) return layerOf[name]
      if (guard[name]) return 0 // defensive only - refs only target earlier steps
      const nextGuard = { ...guard, [name]: true as const }
      const producers = producersOf[name] ?? []
      const layer = producers.length
        ? 1 + Math.max(...producers.map((p) => layerFor(p, nextGuard)))
        : 0
      layerOf[name] = layer
      return layer
    }
    for (const n of nodes) layerFor(n.name, {})

    const columns: FlowNode[][] = []
    for (const n of nodes) {
      const l = layerOf[n.name] ?? 0
      columns[l] = [...(columns[l] ?? []), n]
    }

    // Boxes in a column stack by their own height now - a for_each box
    // holding many chips must not overlap the node beneath it
    const positions: Record<string, { x: number; y: number }> = {}
    const heightOfName: Record<string, number> = {}
    const bottoms: number[] = []
    columns.forEach((col, c) => {
      let y = PAD
      col.forEach((n) => {
        positions[n.name] = { x: PAD + c * COL_W, y }
        heightOfName[n.name] = heightOf(n)
        y += heightOf(n) + ROW_GAP
      })
      bottoms.push(y - ROW_GAP)
    })

    const width = PAD * 2 + BOX_W + Math.max(0, columns.length - 1) * COL_W
    const height = Math.max(...bottoms, PAD * 2 + BOX_H)
    // One clip box per distinct member-box height; the standard box keeps
    // the shared clipPath below
    const tallHeights = [
      ...new Set(nodes.map((n) => heightOf(n)).filter((h) => h > BOX_H)),
    ]

    const edgeLines = edges.map((e) => {
      const from = positions[e.from]
      const to = positions[e.to]
      if (!from || !to) return null
      const x1 = from.x + BOX_W
      const y1 = from.y + (heightOfName[e.from] ?? BOX_H) / 2
      const x2 = to.x
      const y2 = to.y + (heightOfName[e.to] ?? BOX_H) / 2
      // A gentle horizontal-first curve keeps lines readable when an
      // edge skips columns or two edges share a target row.
      const dx = Math.max(40, (x2 - x1) / 2)
      const path = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`
      return { ...e, path, labelX: (x1 + x2) / 2, labelY: (y1 + y2) / 2 }
    })

    return { positions, width, height, tallHeights, edgeLines }
  })

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
    onselect
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
{:else}
  <div class="flowwrap panel">
    <p class="muted hint">
      Read-only data-flow view: boxes are steps, arrows are
      <code>previous_result</code> references labeled with the argument they
      feed. A step with more than one incoming arrow multiplies its inputs
      together (CLAUDE.md's cartesian-product gotcha) - its border is
      highlighted and the multiplier is noted. A step carrying
      <code>for_each</code> shows the entries it runs inset.{#if showsRun}
        A finished step is outlined in green, the one running now in amber
        colour.{/if}{#if onselect}
        Click a step to jump to it in the form view.{/if}
    </p>
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
            <text x={edge.labelX} y={edge.labelY - 6} class="edgelabel"
              >{edge.attribute}</text
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
              class:clickable={!!onselect}
              class:active={node.name === activeStep}
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
                  <g
                    class="member"
                    class:done={state === 'done'}
                    class:active={state === 'active'}
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
                      class="chiplabel">{fit(key, CHIP_CHARS, 'head')}</text
                    >
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
  </div>
{/if}

<style>
  .flowwrap {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .hint code {
    font-family: var(--font-mono);
    font-size: 0.78rem;
  }
  .scrollarea {
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
  .node.fanin .box {
    stroke: var(--warn);
    stroke-width: 2;
  }
  /* Run state last: on a live job which step is running matters more than
     the static fan-in warning the same border would otherwise carry. */
  .node.done .box {
    stroke: var(--good);
    stroke-width: 2;
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
  .entrytag {
    font-size: 9px;
    fill: var(--good);
    text-transform: none;
  }
  .fanlabel {
    font-size: 10px;
    fill: var(--warn);
    font-weight: 600;
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
