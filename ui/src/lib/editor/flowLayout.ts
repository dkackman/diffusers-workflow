/** The geometry of FlowView's diagram: pure functions from the data-flow
 * graph to box positions and edge paths, with the pixel constants the
 * template draws against. Nothing here is reactive - FlowView derives
 * `layoutGraph(graph)` and reads the result. */

import type { DataFlowGraph, FlowEdge, FlowNode } from '../flow'

// Layered left-to-right layout: a node's layer is one past the deepest
// producer that feeds it directly, so entry points (no previous_result
// input) sit in the first column and depth reads as real dependency
// distance rather than just JSON step order.
export const BOX_W = 176
export const BOX_H = 60
const ROW_H = 92
const PAD = 28
// The gap between columns: at least this, wider when an adjacent edge's
// label needs it, so a label sits in the gap rather than under a box
const MIN_GAP = 56
const LABEL_CHARS = 24
const LABEL_CHAR_W = 6.2 // 10px mono
// An edge that skips a column runs in its own lane above every box,
// dropping through the column gaps at each end - a straight line would
// cross the boxes between, and its label land on its neighbours'
const LANE_GAP = 20
const CHANNEL_STEP = 6
const CORNER = 6
// Member chips: a list-driven step's box grows to hold one inset chip
// per entry, beneath the header the ordinary box's three lines occupy
export const CHIP_H = 15
export const CHIP_STEP = 19
export const CHIP_CHARS = 26
export const CHIP_CHARS_TIMED = 17 // a finished chip's run time takes the right end
export const CHIP_TOP = BOX_H - 2
// The gap ROW_H left between fixed-height boxes, kept for the
// height-aware stacking below
const ROW_GAP = ROW_H - BOX_H

// SVG text neither wraps nor takes text-overflow, so a label longer than
// the box ran out of its right edge. Budgets are characters at the box's
// inner width (BOX_W less the 10px inset each side) for each line's font
// - bold 12px sans for the name, 10px mono for the detail - and the
// template's clipPath catches what a wider glyph set still pushes past.
export const NAME_CHARS = 20
export const NAME_CHARS_WITH_TAG = 15 // the entry tag sits in the top-right corner
export const DETAIL_CHARS = 26

/** The text cut to `max` characters with an ellipsis where it was cut: a
 * name is told apart by how it starts, a path by how it ends. */
export function fit(text: string, max: number, keep: 'head' | 'tail'): string {
  if (text.length <= max) return text
  return keep === 'head'
    ? text.slice(0, max - 1) + '…'
    : '…' + text.slice(text.length - max + 1)
}

/** The box's height: the standard header, plus a chip row per entry for
 * a list-driven step - or one empty slot when the step carries
 * for_each but its list cannot be read from the definition. */
export function heightOf(node: FlowNode): number {
  const count = node.members?.length ?? 0
  if (count) return CHIP_TOP + (count - 1) * CHIP_STEP + CHIP_H + 6
  return node.forEach ? BOX_H + 26 : BOX_H
}

/** One line per producer -> consumer pair, carrying every argument it
 * feeds. `lane` is -1 for an edge across one gap, else its index among
 * the column-skipping edges, nearest the boxes first. */
export interface Pair {
  from: string
  to: string
  attributes: string[]
  span: number
  lane: number
}

export interface EdgeLine {
  path: string
  label: string
  title: string
  labelX: number
  labelY: number
}

export interface Point {
  x: number
  y: number
}

export interface FlowLayout {
  positions: Record<string, Point>
  width: number
  height: number
  /** One clip box per distinct member-box height; the standard box keeps
   * the shared clipPath. */
  tallHeights: number[]
  edgeLines: Array<EdgeLine | null>
}

// Plain objects rather than Map/Set throughout: these are throw-away
// scratch structures rebuilt on every recompute, not reactive state, and
// svelte's lint rule wants SvelteMap for anything mutated after
// construction.

/** Each node's layer: one past its deepest direct producer, 0 for an
 * entry point. */
export function layersOf(
  nodes: FlowNode[],
  edges: FlowEdge[],
): Record<string, number> {
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
  return layerOf
}

/** The nodes grouped by layer, JSON order kept within a column. */
export function columnsOf(
  nodes: FlowNode[],
  layerOf: Record<string, number>,
): FlowNode[][] {
  const columns: FlowNode[][] = []
  for (const n of nodes) {
    const l = layerOf[n.name] ?? 0
    columns[l] = [...(columns[l] ?? []), n]
  }
  return columns
}

/** The edges merged by producer -> consumer pair: two references between
 * the same steps drew the same curve twice with their labels printed over
 * each other. Column-skipping pairs get their lanes, shorter spans
 * nearest the boxes so nested lanes never cross. */
export function pairsOf(
  edges: FlowEdge[],
  layerOf: Record<string, number>,
): Pair[] {
  const pairs: Pair[] = []
  for (const e of edges) {
    const pair = pairs.find((p) => p.from === e.from && p.to === e.to)
    if (pair) {
      if (!pair.attributes.includes(e.attribute))
        pair.attributes.push(e.attribute)
    } else
      pairs.push({
        from: e.from,
        to: e.to,
        attributes: [e.attribute],
        span: (layerOf[e.to] ?? 0) - (layerOf[e.from] ?? 0),
        lane: -1,
      })
  }
  const skipping = pairs
    .filter((p) => p.span > 1)
    .sort((a, b) => a.span - b.span)
  skipping.forEach((p, i) => (p.lane = i))
  return pairs
}

function labelOf(p: Pair): string {
  return fit(p.attributes.join(', '), LABEL_CHARS, 'head')
}

/** The column gap: MIN_GAP, or wider when an adjacent edge's label needs it. */
export function gapOf(pairs: Pair[]): number {
  return Math.max(
    MIN_GAP,
    ...pairs
      .filter((p) => p.span === 1)
      .map((p) => labelOf(p).length * LABEL_CHAR_W + 16),
  )
}

interface Placement {
  positions: Record<string, Point>
  heightOfName: Record<string, number>
  bottoms: number[]
}

/** Boxes in a column stack by their own height - a for_each box holding
 * many chips must not overlap the node beneath it. */
function place(columns: FlowNode[][], top: number, colW: number): Placement {
  const positions: Record<string, Point> = {}
  const heightOfName: Record<string, number> = {}
  const bottoms: number[] = []
  columns.forEach((col, c) => {
    let y = top
    col.forEach((n) => {
      positions[n.name] = { x: PAD + c * colW, y }
      heightOfName[n.name] = heightOf(n)
      y += heightOf(n) + ROW_GAP
    })
    bottoms.push(y - ROW_GAP)
  })
  return { positions, heightOfName, bottoms }
}

// Each side of a box spreads its edges down its height rather than
// meeting them at one point. Lane edges take the top ports, the outer
// lane highest, so a lane's vertical never crosses another's stub
function portY(
  pairs: Pair[],
  placed: Placement,
  name: string,
  p: Pair,
  side: 'from' | 'to',
): number {
  const portOrder = (q: Pair, other: string) =>
    q.lane >= 0 ? -1e6 - q.lane : (placed.positions[other]?.y ?? 0)
  const mine = pairs
    .filter((q) => q[side] === name)
    .sort(
      (a, b) =>
        portOrder(a, side === 'from' ? a.to : a.from) -
        portOrder(b, side === 'from' ? b.to : b.from),
    )
  const h = placed.heightOfName[name] ?? BOX_H
  return (
    placed.positions[name].y + (h * (mine.indexOf(p) + 1)) / (mine.length + 1)
  )
}

interface Ends {
  x1: number
  y1: number
  x2: number
  y2: number
}

/** A gentle horizontal-first curve across the one gap. */
function gapCurve(p: Pair, { x1, y1, x2, y2 }: Ends): EdgeLine {
  const dx = Math.max(40, (x2 - x1) / 2)
  return {
    path: `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`,
    label: labelOf(p),
    title: p.attributes.join(', '),
    labelX: (x1 + x2) / 2,
    labelY: (y1 + y2) / 2 - 6,
  }
}

/** Up through the gap after the producer, along the lane, down through
 * the gap before the consumer. The outer lane drops nearest the boxes;
 * past half a gap's worth of lanes they share a channel. */
function laneRoute(
  p: Pair,
  { x1, y1, x2, y2 }: Ends,
  lanes: number,
  gap: number,
  laneY: number,
): EdgeLine {
  const inset = 10 + (lanes - 1 - p.lane) * CHANNEL_STEP
  const xs = Math.min(x1 + inset, x1 + gap / 2 - 2)
  const xd = Math.max(x2 - inset, x2 - gap / 2 + 2)
  const ly = laneY
  const r = CORNER
  const path = [
    `M ${x1} ${y1} H ${xs - r} Q ${xs} ${y1} ${xs} ${y1 - r}`,
    `V ${ly + r} Q ${xs} ${ly} ${xs + r} ${ly}`,
    `H ${xd - r} Q ${xd} ${ly} ${xd} ${ly + r}`,
    `V ${y2 - r} Q ${xd} ${y2} ${xd + r} ${y2} H ${x2}`,
  ].join(' ')
  return {
    path,
    label: labelOf(p),
    title: p.attributes.join(', '),
    labelX: (xs + xd) / 2,
    labelY: ly - 4,
  }
}

function edgeLinesOf(
  pairs: Pair[],
  placed: Placement,
  gap: number,
  top: number,
): Array<EdgeLine | null> {
  const lanes = pairs.filter((p) => p.lane >= 0).length
  return pairs.map((p) => {
    const from = placed.positions[p.from]
    const to = placed.positions[p.to]
    if (!from || !to) return null
    const ends: Ends = {
      x1: from.x + BOX_W,
      y1: portY(pairs, placed, p.from, p, 'from'),
      x2: to.x,
      y2: portY(pairs, placed, p.to, p, 'to'),
    }
    if (p.lane < 0) return gapCurve(p, ends)
    return laneRoute(p, ends, lanes, gap, top - 12 - p.lane * LANE_GAP)
  })
}

/** Where every box and edge of the graph goes. */
export function layoutGraph({ nodes, edges }: DataFlowGraph): FlowLayout {
  const layerOf = layersOf(nodes, edges)
  const columns = columnsOf(nodes, layerOf)
  const pairs = pairsOf(edges, layerOf)
  const lanes = pairs.filter((p) => p.lane >= 0).length
  const top = PAD + lanes * LANE_GAP
  const gap = gapOf(pairs)
  const colW = BOX_W + gap
  const placed = place(columns, top, colW)
  const width = PAD * 2 + BOX_W + Math.max(0, columns.length - 1) * colW
  const height = Math.max(...placed.bottoms, top + PAD + BOX_H)
  const tallHeights = [
    ...new Set(nodes.map((n) => heightOf(n)).filter((h) => h > BOX_H)),
  ]
  return {
    positions: placed.positions,
    width,
    height,
    tallHeights,
    edgeLines: edgeLinesOf(pairs, placed, gap, top),
  }
}
