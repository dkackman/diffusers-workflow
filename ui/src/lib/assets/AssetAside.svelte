<script lang="ts">
  // What the Assets page adds under an asset's recipe in the viewer
  // (RecipePanel's `children`, handed its one read): the job it was saved
  // from, its kind, size and date, the workflows that use it, and the note
  // on it. The note is the listing's, so a saved one rides the listing
  // (the page's `onnotesaved`) and the recipe is not read again
  import type { AssetFile, GalleryMetadata } from '../types'
  import { formatBytes, formatMtime } from '../format'
  import { ASSET } from '../references'
  import { savedFrom } from '../recipe'
  import { wsHref } from '../routes'
  import NoteEditor from '../ui/NoteEditor.svelte'
  import { workspace } from '../workspace.svelte'

  let {
    asset,
    meta,
    onnotesaved,
  }: {
    asset: AssetFile
    /** The panel's read of the asset */
    meta: GalleryMetadata
    /** The note the server stored, so the page can carry it on the asset */
    onnotesaved?: (note: string | null) => void
  } = $props()

  const origin = $derived(savedFrom(meta))
</script>

<div class="asset-aside">
  {#if origin}<p class="muted">{origin}</p>{/if}
  <p class="num muted">
    {asset.kind} · {formatBytes(asset.size)} · {formatMtime(asset.mtime)}
  </p>
  {#if asset.used_by?.length}
    <p class="muted">
      Used by
      {#each asset.used_by as name, i (name)}{i ? ', ' : ''}<a
          href={wsHref(workspace.current, 'workflows', ...name.split('/'))}
        >
          {name}
        </a>{/each}
    </p>
  {/if}
  <NoteEditor
    note={asset.note}
    target={`${ASSET}${asset.name}`}
    label="Your note on this asset"
    onsaved={(note) => onnotesaved?.(note)}
  />
</div>

<style>
  .asset-aside {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }
  p {
    margin: 0;
  }
</style>
