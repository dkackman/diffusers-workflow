<script lang="ts">
  import { Check, CircleDashed, Clock, Lock, Play, X } from '@lucide/svelte'
  import { api } from '../api'
  import { kindGlyph } from '../media'
  import type { LedgerOutput } from '../types'

  // What stands for a job in a request: its picture in a contact-sheet
  // frame, or - while there is none - a same-size dashed slot whose icon
  // says why. The slot is not a .frame: the frame is only ever the picture
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
    /** The take the user accepted: the picked edge of frames.css */
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
  const Icon = $derived(ICONS[shown as keyof typeof ICONS] ?? CircleDashed)
  const picture = $derived(
    output && (output.kind === 'image' || output.kind === 'video')
      ? output
      : null,
  )
  // A result that arrives while the page is open fades in; the one a row
  // is first drawn with does not
  // svelte-ignore state_referenced_locally
  const first = output?.name
  const arrived = $derived(!!output && output.name !== first)
</script>

{#snippet inside()}
  {#if picture}
    {#key picture.name}
      <img
        class:arrived
        src={api.galleryThumbnailUrl(picture.name, workspace)}
        {alt}
        loading="lazy"
      />
    {/key}
    {#if picture.kind === 'video'}
      <span class="videobadge" aria-hidden="true"><Play size={11} /></span>
    {/if}
  {:else if output}
    <!-- Named where it stands alone; a proof inside a named link is not -->
    <span
      class="glyph"
      role={alt ? 'img' : undefined}
      aria-label={alt || undefined}
      aria-hidden={alt ? undefined : 'true'}
    >
      {kindGlyph(output.kind)}
    </span>
  {:else if shown === 'running'}
    <span class="pulse-dot" aria-hidden="true"></span>
  {:else}
    <Icon size={size === 'cover' ? 14 : 18} aria-hidden="true" />
  {/if}
{/snippet}

{#if href}
  <a
    class="plain proof as-{size}"
    class:frame={!!output}
    class:slot={!output}
    class:picked
    {href}
    title="open the job"
  >
    {@render inside()}
  </a>
{:else}
  <span
    class="proof as-{size}"
    class:frame={!!output}
    class:slot={!output}
    class:picked
  >
    {@render inside()}
  </span>
{/if}

<style>
  .proof {
    flex: none;
    width: 96px;
    aspect-ratio: 4 / 3;
  }
  .as-take {
    width: 120px;
  }
  .as-cover {
    width: 48px;
  }
  .as-cover .videobadge {
    top: 2px;
    left: 2px;
    width: 14px;
    height: 14px;
  }
  .slot {
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px dashed var(--line);
    border-radius: var(--radius-frame);
    color: var(--muted);
  }
  a.slot:hover {
    border-color: var(--ink);
    color: var(--ink);
  }
  .frame {
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .glyph {
    font-size: var(--t-md);
    color: var(--muted);
  }
  .as-cover .glyph {
    font-size: var(--t-sm);
  }
  /* The take the user chose: an ink edge, as a ticked tile has */
  .picked {
    border-color: var(--ink);
    box-shadow: inset 0 0 0 1px var(--ink);
  }
  .arrived {
    animation: proof-in 200ms ease-out;
  }
  @keyframes proof-in {
    from {
      opacity: 0;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .arrived {
      animation: none;
    }
  }
</style>
