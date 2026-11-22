<script lang="ts">
  import { bandsOf, toneOf, type Band } from './itemState'

  // A request's plan as a bar. Up to MAX_SEGMENTS items it is one segment
  // an item, in plan order, each a button that goes to its row - equal
  // widths, since an estimate may be missing. Past that, or for a list row
  // that has only counts (`bands`), it is one band a status, as wide as
  // its share. The colour is never alone: each segment is named
  const MAX_SEGMENTS = 40

  type Segment = { key: string; title: string; display: string; word: string }

  let {
    segments,
    bands,
    onselect,
    mini = false,
  }: {
    /** The live items, in plan order */
    segments?: Segment[]
    /** Counts by status, when there are no items to name */
    bands?: Band[]
    onselect?: (key: string) => void
    /** The short bar of a list row */
    mini?: boolean
  } = $props()

  const each = $derived(
    segments && segments.length <= MAX_SEGMENTS ? segments : null,
  )
  const grouped = $derived.by(() => {
    if (bands) return bands
    const counts: Record<string, number> = {}
    for (const s of segments ?? [])
      counts[s.display] = (counts[s.display] ?? 0) + 1
    return bandsOf(counts)
  })
</script>

{#if each}
  <div class="bar" role="group" aria-label="Progress, one segment an item">
    {#each each as segment (segment.key)}
      <button
        type="button"
        class="seg {toneOf(segment.display)}"
        aria-label="{segment.title}: {segment.word}"
        title="{segment.title}: {segment.word}"
        onclick={() => onselect?.(segment.key)}
      ></button>
    {/each}
  </div>
{:else if grouped.length}
  <span
    class="bar"
    class:mini
    role="img"
    aria-label={grouped.map((band) => band.label).join(', ')}
  >
    {#each grouped as band (band.tone)}
      <span
        class="seg {band.tone}"
        style:flex-grow={band.count}
        title={band.label}
      ></span>
    {/each}
  </span>
{/if}

<style>
  .bar {
    display: flex;
    gap: 2px;
    width: 100%;
    max-width: 32rem;
  }
  .mini {
    width: 72px;
    gap: 1px;
  }
  .seg {
    flex: 1 1 0;
    min-width: 3px;
    height: 0.55rem;
    padding: 0;
    border: none;
    border-radius: 2px;
    background: var(--line);
    position: relative;
  }
  .mini .seg {
    height: 4px;
    border-radius: 1px;
  }
  /* A segment is thin; its target is not */
  button.seg::before {
    content: '';
    position: absolute;
    inset: -0.5rem 0;
  }
  .done {
    background: var(--good);
  }
  .running {
    background: var(--live);
  }
  .failed {
    background: var(--bad);
  }
  .queued {
    background: var(--muted);
  }
</style>
