/** A chained pipeline step's `chain` block, summarized for the data-flow
 * graph (flow.ts). The pipeline runs once per segment inside the one step,
 * carrying continuity across each seam, and the segments are stitched into
 * one video (dw/pipeline_processors/chain.py). */

import { VARIABLE, referenceName } from './references'

export interface ChainSummary {
  /** The segment count, when the definition gives it - a literal or a
   * declared variable, which a realized workflow always holds. Null under
   * `match_audio`, or for a variable left unset. */
  segments: number | null
  /** True when the audio's length decides the count at run time. */
  fromAudio: boolean
  /** What each seam carries into the next segment, in a few words. */
  seam: string
  /** True under `anchor: "segment_1"`: segment 1's latents anchor every
   * later segment's (#1166). */
  anchor: boolean
}

// The chain block's engine defaults (dw/workflow_schema.json $defs/chain)
const CHAIN_DEFAULTS: Record<string, unknown> = {
  continuity: 'last_frame',
  trim_frames: 1,
  crossfade_ms: 75,
  guide_frames: 22,
  carry_audio: true,
  anchor: 'none',
}

/** A chain value by key, the engine default standing in for one the block
 * leaves out. */
type ChainRead = (key: string) => unknown

/** The block's reader. A value may be a `variable:`, read from the declared
 * variables - undefined when it names none, or one left null. */
function chainReader(
  workflow: Record<string, any>,
  chain: Record<string, unknown>,
): ChainRead {
  return (key) => {
    const value = key in chain ? chain[key] : CHAIN_DEFAULTS[key]
    const name = referenceName(value, VARIABLE)
    if (name === null) return value
    return workflow.variables?.[name] ?? undefined
  }
}

/** `label`, then the frame count when the block gives a number. */
function withFrames(label: string, frames: unknown): string[] {
  return typeof frames === 'number' ? [label, `${frames} frames`] : [label]
}

function lastFrameSeam(read: ChainRead): string[] {
  const trim = read('trim_frames')
  const fade = read('crossfade_ms')
  return [
    'last frame',
    typeof trim === 'number' ? `trim ${trim}` : null,
    typeof fade === 'number' ? `${fade}ms` : null,
  ].filter((part): part is string => part !== null)
}

/** What the continuity mode carries across the seam, as label words. */
function continuityParts(read: ChainRead): string[] {
  switch (read('continuity')) {
    case 'last_frame':
      return lastFrameSeam(read)
    case 'guide':
      return withFrames('guide', read('guide_frames'))
    case 'last_segment':
      return withFrames('video reference', read('carry_frames'))
    default:
      return ['carry']
  }
}

function seamLabel(read: ChainRead): string {
  const parts = continuityParts(read)
  if (read('continuity') !== 'last_frame' && read('carry_audio') === false)
    parts.push('no audio')
  return parts.join(' · ')
}

/** The declared count, or null when it is unknown here or the audio decides. */
function segmentCount(segments: unknown, matchAudio: unknown): number | null {
  return typeof segments === 'number' && segments >= 1 && !matchAudio
    ? segments
    : null
}

/** The step's `chain` block summarized for the graph, or null without one. */
export function chainSummary(
  workflow: Record<string, any>,
  step: Record<string, any>,
): ChainSummary | null {
  const chain = step.pipeline?.chain
  if (chain === null || typeof chain !== 'object') return null
  const read = chainReader(workflow, chain)
  return {
    segments: segmentCount(read('segments'), chain.match_audio),
    fromAudio: chain.match_audio === true,
    seam: seamLabel(read),
    anchor: read('anchor') === 'segment_1',
  }
}
