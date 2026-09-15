<script lang="ts">
  // The bulk-action chrome a contact sheet with a selection needs: a sticky
  // bar while anything is ticked, and the select-all button above the grid.
  // Shared by the gallery and the assets page, which run the same selection
  // (see picks.svelte.ts) and had two copies of this markup and its CSS.
  import { Download, Trash2 } from '@lucide/svelte'
  import type { Picks } from './picks.svelte'

  let {
    picks,
    visible,
    filterActive,
    busy = false,
    noun,
    ondownload,
    ondelete,
  }: {
    picks: Picks
    /** How many the grid is showing - what "select all" would take. */
    visible: number
    filterActive: boolean
    busy?: boolean
    /** What one row is, for the region label: 'file', 'asset'. */
    noun: string
    ondownload: () => void
    ondelete: () => void
  } = $props()
</script>

{#if picks.size}
  <div class="picks panel" role="region" aria-label="selected {noun}s">
    <strong>{picks.size} selected</strong>
    <span class="flex"></span>
    <button class="withicon" onclick={ondownload} disabled={busy}>
      <Download size={14} />Download .zip
    </button>
    <button class="withicon danger" onclick={ondelete} disabled={busy}>
      <Trash2 size={14} />Delete
    </button>
    <button class="quiet" onclick={() => picks.clear()} disabled={busy}
      >Clear</button
    >
  </div>
{/if}

<div class="picktools">
  <button
    class="quiet"
    onclick={() => picks.selectAll()}
    disabled={visible === 0}
    >Select all{filterActive ? ' matching' : ''} ({visible})</button
  >
</div>

<style>
  .picks {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.8rem;
    margin-bottom: 0.6rem;
    position: sticky;
    top: 0;
    z-index: 2;
  }
  .flex {
    flex: 1;
  }
  .picktools {
    margin-bottom: 0.6rem;
  }
  .withicon {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
</style>
