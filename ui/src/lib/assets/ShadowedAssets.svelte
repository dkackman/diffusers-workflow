<script lang="ts">
  import { collapsedState } from '../collapsed.svelte'
  import { leafName } from '../names'
  import GroupHeading from '../ui/GroupHeading.svelte'
  import OutputTile from '../ui/OutputTile.svelte'
  import type { MediaFile } from '../media'
  import type { AssetFile, ShadowedAsset } from '../types'

  // The entries a library holds under a name a nearer library has taken -
  // a dimmed label per entry rather than a picture, since the server serves
  // the file that won
  let {
    entries,
    filterActive = false,
  }: { entries: ShadowedAsset[]; filterActive?: boolean } = $props()

  // A name, not a file: no URL and no thumbnail, since the server would
  // serve the file that won
  const ghost = (entry: ShadowedAsset): MediaFile => ({
    name: entry.name,
    kind: entry.kind,
    url: '',
    label: leafName(entry.name),
    workspace: '',
    thumb: null,
  })

  const collapsed = collapsedState('collapsed-asset-shadowed')
  const library = $derived(entries[0]?.origin ?? '')
  // While filtering everything stays open: a shut section would hide matches
  const shut = $derived(!filterActive && collapsed.has(library))

  // The same libraries in the possessive, for "shadowed by ...". Built by
  // hand rather than off the labels: "the examples's" is not English
  const SHADOWED_BY: Record<AssetFile['origin'], string> = {
    workspace: "this workspace's",
    common: "the shared asset library's",
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
    collapsed={shut}
    disabled={filterActive}
    ontoggle={() => collapsed.toggle(library)}
    title={filterActive
      ? 'Showing every match while the filter is on'
      : shut
        ? 'expand the shadowed entries'
        : 'collapse the shadowed entries'}
  />
</div>
{#if !shut}
  <div class="grid">
    {#each entries as entry (entry.name)}
      <!-- role + aria-label, not title alone: a bare div is not exposed,
           and the title is the only thing that explains why this tile is
           here -->
      <div
        class="cellwrap shadowed"
        role="note"
        aria-label="shadowed by {SHADOWED_BY[
          entry.shadowed_by
        ]} {entry.name} - {entry.reference} resolves to that file"
        title="shadowed by {SHADOWED_BY[
          entry.shadowed_by
        ]} {entry.name} - {entry.reference} resolves to that file"
      >
        <OutputTile file={ghost(entry)} dim />
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
</style>
