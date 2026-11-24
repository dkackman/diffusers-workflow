<script lang="ts">
  // One file in the Gallery's grid: its tick for the bulk actions, and its
  // tile, which opens it in the viewer
  import type { MediaFile } from '../media'
  import type { Picks } from '../picks.svelte'
  import type { GalleryFile } from '../types'
  import OutputTile from '../ui/OutputTile.svelte'
  import { versionTitle } from '../terms'
  import RoleTag from './RoleTag.svelte'

  let {
    file,
    tile,
    picks,
    changed,
    active,
    onopen,
  }: {
    file: GalleryFile
    tile: MediaFile
    picks: Picks
    /** Marked as another writer's change (resource's `changed`) */
    changed: boolean
    /** Open in the viewer */
    active: boolean
    onopen: () => void
  } = $props()
  const name = $derived(file.name)
</script>

<div
  class="cellwrap"
  class:picked={picks.has(name)}
  class:row-changed={changed}
>
  <!-- A sibling of the cell rather than a child: a checkbox nested in
   a button is invalid, and keeping them apart is what lets a plain
   click still open the details it always has -->
  <input
    class="pick"
    type="checkbox"
    checked={picks.has(name)}
    aria-label="select {name}"
    title="select this file for a bulk action"
    onclick={(e) => picks.toggle(name, e.shiftKey)}
  />
  <OutputTile
    file={tile}
    caption="clamp"
    {active}
    picked={picks.has(name)}
    onclick={onopen}
    title="Open this file, with what made it"
  >
    {#snippet badges()}
      <RoleTag role={file.role} />
      {#if file.version}
        <span class="version" title={versionTitle(file.version)}>
          v{file.version}
        </span>
      {/if}
    {/snippet}
  </OutputTile>
</div>

<style>
  /* The one mark that tells four runs' same-named files apart */
  .version {
    display: inline-block;
    padding: 0 0.3rem;
    border-radius: 0.2rem;
    background: var(--line);
    color: var(--ink);
    font-weight: 600;
  }
</style>
