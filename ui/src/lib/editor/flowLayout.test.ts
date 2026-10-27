import { describe, expect, it } from 'vitest'
import { dataFlowGraph } from '../flow'
import { BOX_W, layersOf, layoutGraph, pairsOf } from './flowLayout'

const step = (name: string, args: Record<string, string> = {}) => ({
  name,
  pipeline: { configuration: { component_type: 'Fake' }, arguments: args },
})

describe('layoutGraph', () => {
  it('merges two references between the same steps into one labelled edge', () => {
    const graph = dataFlowGraph({
      steps: [
        step('gen'),
        step('edit', {
          image: 'previous_result:gen',
          mask: 'previous_result:gen',
        }),
      ],
    })
    const pairs = pairsOf(graph.edges, layersOf(graph.nodes, graph.edges))
    expect(pairs).toHaveLength(1)
    expect(pairs[0].attributes).toEqual(['image', 'mask'])
    expect(pairs[0].lane).toBe(-1)
    const layout = layoutGraph(graph)
    expect(layout.edgeLines).toHaveLength(1)
    expect(layout.edgeLines[0]?.label).toBe('image, mask')
    expect(layout.edgeLines[0]?.title).toBe('image, mask')
  })

  it('places a consumer one column right of its deepest producer', () => {
    const graph = dataFlowGraph({
      steps: [
        step('a'),
        step('b', { image: 'previous_result:a' }),
        step('c', { image: 'previous_result:b', mask: 'previous_result:a' }),
      ],
    })
    expect(layersOf(graph.nodes, graph.edges)).toEqual({ a: 0, b: 1, c: 2 })
    const { positions } = layoutGraph(graph)
    expect(positions.b.x - positions.a.x).toBeGreaterThan(BOX_W)
    expect(positions.c.x - positions.b.x).toBe(positions.b.x - positions.a.x)
  })

  it('routes a column-skipping edge in a lane above the boxes', () => {
    const graph = dataFlowGraph({
      steps: [
        step('a'),
        step('b', { image: 'previous_result:a' }),
        step('c', { image: 'previous_result:b', mask: 'previous_result:a' }),
      ],
    })
    const pairs = pairsOf(graph.edges, layersOf(graph.nodes, graph.edges))
    const skip = pairs.find((p) => p.from === 'a' && p.to === 'c')!
    expect(skip.span).toBe(2)
    expect(skip.lane).toBe(0)
    const layout = layoutGraph(graph)
    const laneEdge = layout.edgeLines[pairs.indexOf(skip)]!
    // The lane sits above every box, and the label on it
    expect(laneEdge.labelY).toBeLessThan(layout.positions.a.y)
    expect(laneEdge.path).toContain(' V ')
    const direct = layout.edgeLines[pairs.indexOf(pairs[0])]!
    expect(direct.path.startsWith('M ')).toBe(true)
    expect(direct.path).toContain(' C ')
  })
})
