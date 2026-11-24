<script lang="ts">
  // What the Assets page adds beside an asset's recipe in the viewer: its
  // kind, size and date, the workflows that use it, and the note on it.
  // None of it needs the recipe read, so it shows while that loads or
  // fails. The note is the listing's, so a saved one rides the listing
  // (the page's `onnotesaved`) and the recipe is not read again
  import type { AssetFile } from '../types'
  import { formatBytes, formatMtime } from '../format'
  import { ASSET } from '../references'
  import { wsHref } from '../routes'
  import NoteEditor from '../ui/NoteEditor.svelte'
  import { workspace } from '../workspace.svelte'

  let {
    asset,
    onnotesaved,
  }: {
    asset: AssetFile
    /** The note the server stored, so the page can carry it on the asset */
    onnotesaved?: (note: string | null) => void
  } = $props()
</script>

<div class="asset-aside">
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
