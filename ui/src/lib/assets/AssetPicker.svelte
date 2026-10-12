<script lang="ts">
  // Browse the asset library and pick one: the grid the Assets page shows,
  // in a dialog over the field that wants a name. Typing asset: in a field
  // suits someone who knows the name; this suits picking by eye.
  import Modal from '../ui/Modal.svelte'
  import { assetThumbnailUrl } from '../apiUrls'
  import { assetLibrary, ensureAssetLibrary } from '../assetlib.svelte'
  import { hasThumbnail } from '../refcomplete.svelte'
  import { leafName } from '../names'
  import { matchesWords } from '../ui/suggest'
  import type { AssetFile } from '../types'

  let {
    open = $bindable(false),
    kind = undefined,
    current = null,
    onpick,
  }: {
    open?: boolean
    /** Only this kind of media, for a field that takes one */
    kind?: AssetFile['kind']
    /** The asset the field holds now, marked in the grid */
    current?: string | null
    /** The chosen asset's name, without the asset: prefix */
    onpick: (name: string) => void
  } = $props()

  const LIBRARY_LABELS: Record<AssetFile['origin'], string> = {
    workspace: 'This workspace',
    common: 'Shared library',
    examples: 'Examples',
  }
  const ORDER: AssetFile['origin'][] = ['workspace', 'common', 'examples']

  let query = $state('')

  $effect(() => {
    if (open) ensureAssetLibrary()
  })

  const matching = $derived(
    (assetLibrary.assets ?? []).filter(
      (asset) =>
        (!kind || asset.kind === kind) && matchesWords(query, asset.name),
    ),
  )
  // By library, in the order a run resolves names: what is this
  // workspace's own comes first
  const sections = $derived(
    ORDER.map((origin) => ({
      origin,
      label: LIBRARY_LABELS[origin],
      assets: matching
        .filter((asset) => asset.origin === origin)
        .sort((a, b) => a.name.localeCompare(b.name)),
    })).filter((section) => section.assets.length),
  )

  function pick(asset: AssetFile) {
    onpick(asset.name)
    open = false
    query = ''
  }
</script>

<Modal bind:open label="choose an asset" large>
  <div class="head">
    <h2>Choose {kind === 'image' ? 'an image' : `a ${kind ?? 'file'}`}</h2>
    <input
      class="search"
      placeholder="search…"
      aria-label="search the asset library"
      bind:value={query}
    />
  </div>
  <div class="body">
    {#if assetLibrary.assets === undefined || assetLibrary.workspace === undefined}
      <p class="muted">Loading the asset library…</p>
    {:else if sections.length === 0}
      <p class="muted">
        {query
          ? `No ${kind ?? 'asset'} matches "${query}".`
          : `No ${kind ?? 'asset'}s in the library yet.`}
      </p>
    {/if}
    {#each sections as section (section.origin)}
      <h3 class="library">
        {section.label} <span class="muted">({section.assets.length})</span>
      </h3>
      <div class="grid">
        {#each section.assets as asset (asset.name)}
          <button
            class="tile"
            class:current={asset.name === current}
            aria-current={asset.name === current ? 'true' : undefined}
            onclick={() => pick(asset)}
            title={asset.reference}
            aria-label="choose {asset.name}"
          >
            {#if hasThumbnail(asset)}
              <img src={assetThumbnailUrl(asset.name)} alt="" loading="lazy" />
            {:else}
              <span class="glyph" aria-hidden="true"
                >{asset.kind === 'audio' ? '♪' : '¶'}</span
              >
            {/if}
            <span class="caption">{leafName(asset.name)}</span>
            {#if asset.folder}<span class="folder muted">{asset.folder}</span
              >{/if}
          </button>
        {/each}
      </div>
    {/each}
  </div>
  <div class="ui-actions">
    <button class="quiet" onclick={() => (open = false)}>Cancel</button>
  </div>
</Modal>

<style>
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2) var(--space-4);
    margin-bottom: var(--space-3);
  }
  .head h2 {
    margin: 0;
  }
  .search {
    flex: 1;
    min-width: 12rem;
  }
  /* The grid scrolls; the search and Cancel stay put */
  .body {
    flex: 1;
    min-height: 8rem;
    overflow-y: auto;
    margin-bottom: var(--space-3);
  }
  .library {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
    margin: var(--space-3) 0 var(--space-2);
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: var(--space-2);
  }
  .tile {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    padding: 0 0 var(--space-1);
    border: 1px solid var(--line);
    border-radius: var(--radius-frame);
    background: var(--panel);
    color: var(--ink);
    overflow: hidden;
    text-align: left;
    cursor: pointer;
  }
  .tile:hover,
  .tile:focus-visible {
    border-color: var(--accent);
    filter: none;
  }
  .tile.current {
    border-color: var(--accent);
    box-shadow: inset 0 0 0 1px var(--accent);
  }
  .tile img,
  .glyph {
    display: block;
    width: 100%;
    height: 96px;
    object-fit: cover;
    background: var(--sunk);
  }
  .glyph {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    color: var(--muted);
  }
  .caption,
  .folder {
    padding: 0 var(--space-2);
    font-family: var(--font-mono);
    font-size: 0.75rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .folder {
    font-size: 0.7rem;
  }
</style>
