<script lang="ts">
  import { dataFlowGraph, type FlowNode } from '../flow'

  let {
    workflow,
    onselect = undefined,
  }: {
    workflow: Record<string, any>
    onselect?: (stepName: string) => void
  } = $props()

  const graph = $derived(dataFlowGraph(workflow))

  // Layered left-to-right layout: a node's layer is one past the deepest
  // producer that feeds it directly, so entry points (no previous_result
  // input) sit in the first column and depth reads as real dependency
  // distance rather than just JSON step order.
  const BOX_W = 176
  const BOX_H = 60
  const COL_W = 232
  const ROW_H = 92
  const PAD = 28

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

    const positions: Record<string, { x: number; y: number }> = {}
    columns.forEach((col, c) => {
      col.forEach((n, r) => {
        positions[n.name] = { x: PAD + c * COL_W, y: PAD + r * ROW_H }
      })
    })

    const maxRows = Math.max(1, ...columns.map((c) => c.length))
    const width = PAD * 2 + BOX_W + Math.max(0, columns.length - 1) * COL_W
    const height = PAD * 2 + BOX_H + (maxRows - 1) * ROW_H

    const edgeLines = edges.map((e) => {
      const from = positions[e.from]
      const to = positions[e.to]
      if (!from || !to) return null
      const x1 = from.x + BOX_W
      const y1 = from.y + BOX_H / 2
      const x2 = to.x
      const y2 = to.y + BOX_H / 2
      // A gentle horizontal-first curve keeps lines readable when an
      // edge skips columns or two edges share a target row.
      const dx = Math.max(40, (x2 - x1) / 2)
      const path = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`
      return { ...e, path, labelX: (x1 + x2) / 2, labelY: (y1 + y2) / 2 }
    })

    return { positions, width, height, edgeLines }
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
</script>

{#if (graph.nodes ?? []).length === 0}
  <p class="muted">No steps yet.</p>
{:else}
  <div class="flowwrap panel">
    <p class="muted hint">
      Read-only data-flow view: boxes are steps, arrows are
      <code>previous_result</code> references labeled with the argument they feed.
      A step with more than one incoming arrow multiplies its inputs together (CLAUDE.md's
      cartesian-product gotcha) - its border is highlighted and the multiplier is
      noted. Click a step to jump to it in the form view.
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
            <g
              class="node"
              class:entry={node.isEntryPoint}
              class:fanin={!!fanIn}
              transform={`translate(${pos.x}, ${pos.y})`}
              role="button"
              tabindex="0"
              aria-label={`step ${node.name}, ${kindLabel(node.kind)}${node.isEntryPoint ? ', entry point' : ''}${fanIn ? ', fan-in: ' + fanIn.label : ''}`}
              onclick={() => select(node.name)}
              onkeydown={(e) => onKeydown(e, node.name)}
            >
              <rect width={BOX_W} height={BOX_H} rx="8" class="box" />
              <text x="10" y="20" class="stepname">{node.name}</text>
              <text x="10" y="37" class="stepkind">{kindLabel(node.kind)}</text>
              {#if node.detail}
                <text x="10" y="52" class="stepdetail">{node.detail}</text>
              {/if}
              {#if node.isEntryPoint}
                <text x={BOX_W - 8} y="14" class="entrytag" text-anchor="end"
                  >entry</text
                >
              {/if}
            </g>
            {#if fanIn}
              <text
                x={pos.x + BOX_W / 2}
                y={pos.y + BOX_H + 14}
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
    font-family: ui-monospace, 'Cascadia Code', monospace;
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
    font-family: ui-monospace, 'Cascadia Code', monospace;
  }
  .node {
    cursor: pointer;
  }
  .box {
    fill: var(--panel);
    stroke: var(--line);
    stroke-width: 1.5;
  }
  .node:hover .box,
  .node:focus-visible .box {
    stroke: var(--accent);
  }
  .node.entry .box {
    stroke-dasharray: 4 3;
  }
  .node.fanin .box {
    stroke: var(--warn);
    stroke-width: 2;
  }
  .stepname {
    font-size: 12px;
    font-weight: 700;
    fill: var(--ink);
  }
  .stepkind {
    font-size: 10px;
    fill: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .stepdetail {
    font-size: 10px;
    fill: var(--muted);
    font-family: ui-monospace, 'Cascadia Code', monospace;
  }
  .entrytag {
    font-size: 9px;
    fill: var(--good);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .fanlabel {
    font-size: 10px;
    fill: var(--warn);
    font-weight: 600;
  }
</style>
