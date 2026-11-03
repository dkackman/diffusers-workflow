<svelte:options namespace="svg" />

<!-- The segments a chained step runs, drawn inside its node box: a summary
     line, one chip per segment in order, each seam's carry written between
     them. Drawn in the SVG namespace, so it sits inside FlowView's <g>. -->
<script lang="ts">
  import type { FlowNode } from '../flow'
  import {
    BOX_W,
    CHAIN_SUMMARY_Y,
    CHIP_H,
    SEAM_CHARS,
    SEGMENT_STEP,
    SEGMENT_TOP,
    fit,
    segmentChips,
  } from './flowLayout'

  let {
    node,
    stepState = '',
    activeSegment = undefined,
  }: {
    /** A node with a `chain` - the caller draws nothing else through here. */
    node: FlowNode
    /** The step's run state: 'active', 'failed', 'done', 'pending' or ''. */
    stepState?: string
    /** The segment, 1-based, the run is on or last ran - its generating
     * phases (runstate.ts chainSegments). */
    activeSegment?: number
  } = $props()

  const label = $derived.by(() => {
    if (!node.chain) return ''
    if (node.chain.fromAudio) return 'chain · length from audio'
    const n = node.chain.segments
    return n === null ? 'chain' : `chain · ${n} segment${n === 1 ? '' : 's'}`
  })
  const chips = $derived(segmentChips(node))

  // A segment's state follows its step's: every segment of a finished step
  // is done, and on a running or failed one, those before the segment it
  // is on are done and that one carries the step's state
  const segmentStateOf = $derived((segment: number) => {
    if (stepState === 'done') return 'done'
    if (stepState !== 'active' && stepState !== 'failed') return ''
    if (activeSegment === undefined) return ''
    return segment < activeSegment
      ? 'done'
      : segment === activeSegment
        ? stepState
        : ''
  })
</script>

<text x="11" y={CHAIN_SUMMARY_Y} class="chainsummary">{label}</text>
{#each Array.from({ length: chips }, (_, i) => i) as i (i)}
  {@const y = SEGMENT_TOP + i * SEGMENT_STEP}
  {@const state = segmentStateOf(i + 1)}
  <g
    class="segment"
    class:done={state === 'done'}
    class:active={state === 'active'}
    class:failed={state === 'failed'}
    aria-label={`segment ${i + 1}${state ? ', ' + state : ''}`}
  >
    <rect x="10" {y} width={BOX_W - 20} height={CHIP_H} rx="4" class="chip" />
    <text x="15" y={y + 11} class="chiplabel">segment {i + 1}</text>
    {#if i === 0 && node.chain?.anchor}
      <g>
        <title
          >segment 1's latents anchor every later segment's (chain anchor:
          segment_1)</title
        >
        <text x={BOX_W - 15} y={y + 11} class="anchortag" text-anchor="end"
          >anchor</text
        >
      </g>
    {/if}
  </g>
  {#if i < chips - 1 && node.chain}
    <text x="15" y={y + CHIP_H + 10} class="seamlabel"
      >{#if `↓ ${node.chain.seam}`.length > SEAM_CHARS}<title
          >{node.chain.seam}</title
        >{/if}{fit(`↓ ${node.chain.seam}`, SEAM_CHARS, 'head')}</text
    >
  {/if}
{/each}

<style>
  /* Machine state on the chips' edges, as on the nodes: done green, running
     the safelight amber (FlowView.svelte's .member chips match). */
  .segment .chip {
    fill: var(--panel-2);
    stroke: var(--line);
    stroke-width: 1;
  }
  .segment.done .chip {
    stroke: var(--good);
    stroke-width: 1.5;
  }
  .segment.failed .chip {
    stroke: var(--bad);
    stroke-width: 1.5;
  }
  .segment.active .chip {
    stroke: var(--live);
    stroke-width: 1.5;
    animation: dw-pulse 1.6s ease-in-out infinite;
  }
  @media (prefers-reduced-motion: reduce) {
    .segment.active .chip {
      animation: none;
    }
  }
  .chiplabel {
    font-size: 9px;
    fill: var(--ink);
    font-family: var(--font-mono);
  }
  .chainsummary,
  .seamlabel,
  .anchortag {
    font-size: 9px;
    fill: var(--muted);
    font-family: var(--font-mono);
  }
</style>
