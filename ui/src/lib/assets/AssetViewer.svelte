<script lang="ts">
  // The Assets page's viewer: the asset large, its reference to copy, a
  // download and a delete (none for a read-only examples asset), and
  // beside it what made it (RecipePanel, its one read) with AssetAside
  // after it, which does not wait for that read. The page hands in what the viewer steps through and keeps
  // the open asset in the hash (mediaViewer.ts)
  import { Trash2 } from '@lucide/svelte'
  import DownloadLink from '../DownloadLink.svelte'
  import type { MediaFile } from '../media'
  import { ASSET } from '../references'
  import { savedFrom } from '../recipe'
  import type { AssetFile } from '../types'
  import MediaViewer from '../ui/MediaViewer.svelte'
  import RecipePanel from '../ui/RecipePanel.svelte'
  import ReferenceChip from '../ui/ReferenceChip.svelte'
  import AssetAside from './AssetAside.svelte'

  let {
    items,
    open = $bindable(null),
    assetOf,
    busy,
    onremove,
    onnotesaved,
    ongone,
  }: {
    items: MediaFile[]
    open: string | null
    /** The listed asset by name, as the page holds it (with its notes) */
    assetOf: (name: string) => AssetFile | undefined
    busy: boolean
    onremove: (asset: AssetFile) => void
    /** The note the server stored on an asset */
    onnotesaved: (name: string, note: string | null) => void
    /** A file's read came back 404 (MediaViewer's `ongone`) */
    ongone?: (name: string) => void
  } = $props()
</script>

{#snippet actions(file: MediaFile)}
  {@const asset = assetOf(file.name)}
  <ReferenceChip prefix={ASSET} name={file.name} />
  <DownloadLink href={file.url} />
  {#if asset?.origin === 'examples'}
    <span class="muted" title="Read-only: an examples library brought it">
      Read-only
    </span>
  {:else if asset}
    <button
      type="button"
      class="quiet icon danger"
      onclick={() => onremove(asset)}
      disabled={busy}
      aria-label="Delete"
      title="Delete this asset from the library"
    >
      <Trash2 size={16} />
    </button>
  {/if}
{/snippet}

{#snippet aside(file: MediaFile)}
  {@const asset = assetOf(file.name)}
  <!-- Keyed by the asset, so each starts its own read and its own note box -->
  {#key file.name}
    <RecipePanel name={file.name} source="asset" workspace={file.workspace}>
      {#snippet children(meta)}
        {@const origin = savedFrom(meta)}
        {#if origin}<p class="muted origin">{origin}</p>{/if}
      {/snippet}
    </RecipePanel>
    {#if asset}
      <AssetAside
        {asset}
        onnotesaved={(note) => onnotesaved(asset.name, note)}
      />
    {/if}
  {/key}
{/snippet}

<MediaViewer {items} bind:open {actions} {aside} {ongone} />

<style>
  .origin {
    margin: 0;
  }
</style>
