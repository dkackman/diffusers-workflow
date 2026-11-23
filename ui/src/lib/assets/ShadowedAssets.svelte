<script lang="ts">
  import { collapsedState } from '../collapsed.svelte'
  import { leafName } from '../names'
  import GroupHeading from '../ui/GroupHeading.svelte'
  import type { AssetFile, ShadowedAsset } from '../types'

  // The entries a library holds under a name a nearer library has taken -
  // a dimmed label per entry rather than a picture, since the server serves
  // the file that won
  let { entries }: { entries: ShadowedAsset[] } = $props()

  const collapsed = collapsedState('collapsed-asset-shadowed')
  const library = $derived(entries[0]?.origin ?? '')

  // The same libraries in the possessive, for "shadowed by ...". Built by
  // hand rather than off the labels: "the examples's" is not English
  const SHADOWED_BY: Record<AssetFile['origin'], string> = {
    workspace: "this workspace's",
    common: "the shared library's",
    examples: "the examples library's",
  }
</script>

<!-- What this library holds under a name a nearer one has taken. It
     is here because "I uploaded it and asset: still loads the old
     one" is otherwise unanswerable from the page - the server does
     not serve these, so a tile is a dimmed label rather than a
     picture, and nothing bulk can reach it -->
<div class="grouprow">
  <GroupHeading
    label="shadowed/"
    count={entries.length}
    collapsed={collapsed.has(library)}
    ontoggle={() => collapsed.toggle(library)}
    title={collapsed.has(library)
      ? 'expand the shadowed entries'
      : 'collapse the shadowed entries'}
  />
</div>
{#if !collapsed.has(library)}
  <div class="grid">
    {#each entries as entry (entry.name)}
      <div class="cellwrap">
        <!-- role + aria-label, not title alone: a bare div is not
           exposed, and the title is the only thing that explains
           why this tile is here -->
        <div
          class="cell shadowed"
          role="note"
          aria-label="shadowed by {SHADOWED_BY[
            entry.shadowed_by
          ]} {entry.name} - {entry.reference} resolves to that file"
          title="shadowed by {SHADOWED_BY[
            entry.shadowed_by
          ]} {entry.name} - {entry.reference} resolves to that file"
        >
          <span class="ghost">{entry.kind}</span>
          <span class="caption">{leafName(entry.name)}</span>
        </div>
      </div>
    {/each}
  </div>
{/if}

<style>
  .grouprow {
    margin: 1.2rem 0 var(--space-2);
  }
  .grid {
    display: grid;
    align-items: start;
    gap: 0.6rem;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  }
  /* The frame is the app's one picture surface: media edge to edge, the
     size set here rather than on the media */
  .cell {
    display: block;
    width: 100%;
    padding: 0;
    border: 1px solid var(--line);
    border-radius: var(--radius-frame);
    background: var(--panel);
    overflow: hidden;
    cursor: pointer;
    text-align: left;
    /* As in the gallery: every cell's DOM node is still rendered, but
       content-visibility skips layout and paint for the ones scrolled out
       of view, which is most of the win for a large library */
    content-visibility: auto;
    contain-intrinsic-size: 150px 134px;
  }
  .cell:hover {
    border-color: var(--accent);
    filter: none;
  }
  /* The shadowed tile has no picture to show, and has to keep the grid's
     rhythm as the audio placeholder does */
  .cell .ghost {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 110px;
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: var(--t-xs);
    padding: 0 0.4rem;
    overflow: hidden;
  }
  .caption {
    display: block;
    padding: 0.3rem 0.45rem;
    font-family: var(--font-mono);
    font-size: var(--t-xs);
    color: var(--ink);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  /* A shadowed entry is a name, not a file: the server serves the tile
     that won, so there is nothing to show and nothing to do with it */
  .cell.shadowed {
    opacity: 0.45;
    cursor: default;
  }
</style>
