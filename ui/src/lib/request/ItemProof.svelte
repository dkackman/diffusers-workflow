<script lang="ts">
  import { Check, CircleDashed, Clock, Lock, X } from '@lucide/svelte'
  import { ledgerMedia } from '../media'
  import type { LedgerOutput } from '../types'
  import OutputTile from '../ui/OutputTile.svelte'

  // What stands for a job in a request: its file as a tile, or - while
  // there is none - a same-size dashed slot whose icon says why. The slot is
  // not a tile: the tile is only ever the file
  let {
    output,
    state: shown,
    workspace,
    href = null,
    alt,
    size = 'row',
    picked = false,
  }: {
    /** The job's `output` as the ledger gives it; null when it wrote none */
    output: LedgerOutput | null
    /** The item's display state or the attempt's status: picks the slot's icon */
    state: string
    workspace: string
    /** The job it leads to; a proof inside a link passes none */
    href?: string | null
    alt: string
    size?: 'row' | 'take' | 'cover'
    /** The take the user accepted: the tile's picked edge */
    picked?: boolean
  } = $props()

  const ICONS = {
    waiting: Lock,
    queued: Clock,
    failed: X,
    cancelled: X,
    done: Check,
    succeeded: Check,
  }
  const TILE_SIZES = { row: 'strip', take: 'take', cover: 'mini' } as const
  const Icon = $derived(ICONS[shown as keyof typeof ICONS] ?? CircleDashed)
  // A result that arrives while the page is open fades in; the one a row
  // is first drawn with does not
  // svelte-ignore state_referenced_locally
  const first = output?.name
  const arrived = $derived(!!output && output.name !== first)
</script>

{#if output}
  {#key output.name}
    <!-- Named where it stands alone; a proof inside a named link is not -->
    <span class="tilewrap as-{size}" class:arrived>
      <OutputTile
        file={ledgerMedia(output, workspace)}
        size={TILE_SIZES[size]}
        caption="none"
        href={href ?? undefined}
        {alt}
        {picked}
        title={href ? 'open the job' : undefined}
        aria-label={href && alt ? alt : undefined}
      />
    </span>
  {/key}
{:else if href}
  <a class="plain slot as-{size}" {href} title="open the job">
    {@render why()}
  </a>
{:else}
  <span class="slot as-{size}">{@render why()}</span>
{/if}

{#snippet why()}
  {#if shown === 'running'}
    <span class="pulse-dot" aria-hidden="true"></span>
  {:else}
    <Icon size={size === 'cover' ? 14 : 18} aria-hidden="true" />
  {/if}
{/snippet}

<style>
  /* The slot takes the frame of the tile it stands in for */
  .slot {
    flex: none;
    width: 96px;
    aspect-ratio: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px dashed var(--line);
    border-radius: var(--radius-frame);
    color: var(--muted);
  }
  .slot.as-take {
    width: 240px;
    aspect-ratio: 4 / 3;
  }
  .slot.as-cover {
    width: 48px;
  }
  a.slot:hover {
    border-color: var(--ink);
    color: var(--ink);
  }
  .tilewrap {
    flex: none;
    display: flex;
  }
  .as-cover :global(.videobadge) {
    top: 2px;
    left: 2px;
    width: 14px;
    height: 14px;
  }
  .arrived :global(.tile-pic) {
    animation: proof-in 200ms ease-out;
  }
  @keyframes proof-in {
    from {
      opacity: 0;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .arrived :global(.tile-pic) {
      animation: none;
    }
  }
</style>
