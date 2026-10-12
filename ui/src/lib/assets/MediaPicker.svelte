<script lang="ts">
  // Browse for a file a field can name and pick one by eye: the asset
  // library, or what earlier runs wrote. Typing asset: or output: in a field
  // suits someone who knows the name; this suits picking by what it shows.
  import { untrack } from 'svelte'
  import Modal from '../ui/Modal.svelte'
  import { assetThumbnailUrl, galleryThumbnailUrl } from '../apiUrls'
  import { assetLibrary, ensureAssetLibrary } from '../assetlib.svelte'
  import {
    ensureOutputLibrary,
    latestName,
    loadOutputLibrary,
    outputLibrary,
    outputNamed,
  } from '../outputlib.svelte'
  import { hasThumbnail } from '../refcomplete.svelte'
  import { ASSET, OUTPUT, reference, referenceName } from '../references'
  import { leafName } from '../names'
  import { matchesWords } from '../ui/suggest'
  import type { AssetFile } from '../types'

  type Tab = 'assets' | 'outputs'

  let {
    open = $bindable(false),
    kind = undefined,
    current = '',
    onpick,
  }: {
    open?: boolean
    /** Only this kind of media, for a field that takes one */
    kind?: AssetFile['kind']
    /** The field's value now: its asset or output is marked, and an
     * output: one opens on the runs tab */
    current?: unknown
    /** The chosen file, as the reference the field takes */
    onpick: (value: string) => void
  } = $props()

  const LIBRARY_LABELS: Record<AssetFile['origin'], string> = {
    workspace: 'This workspace',
    common: 'Shared library',
    examples: 'Examples',
  }
  const ORDER: AssetFile['origin'][] = ['workspace', 'common', 'examples']
  const GLYPHS: Record<string, string> = { audio: '♪' }

  const currentAsset = $derived(referenceName(current, ASSET))
  const currentOutput = $derived(referenceName(current, OUTPUT))

  let tab = $state<Tab>('assets')
  let query = $state('')
  // `latest` in the run's place: a later stage keeps naming the newest
  // product without being edited after every run
  let followLatest = $state(false)

  // Each opening starts where the field's own value lives
  $effect(() => {
    if (!open) return
    tab = currentOutput !== null ? 'outputs' : 'assets'
    followLatest = currentOutput?.split('/').includes('latest') ?? false
  })
  $effect(() => {
    if (!open) return
    if (tab === 'assets') ensureAssetLibrary()
    else ensureOutputLibrary()
  })
  // Runs finish all the time: each opening fetches what they wrote, so the
  // newest is there even when the listing loaded earlier this session
  $effect(() => {
    if (!open) return
    // Read untracked: the effect is about the opening, not the listing
    untrack(() => {
      if (outputLibrary.workspace !== undefined) void loadOutputLibrary()
    })
  })

  type Tile = {
    key: string
    caption: string
    sub: string
    thumb?: string
    glyph: string
    current: boolean
    pick: () => string
  }
  type Section = { key: string; label: string; tiles: Tile[] }

  const fits = (fileKind: string) => !kind || fileKind === kind

  // By library, in the order a run resolves names: what is this
  // workspace's own comes first
  const assetSections = $derived.by((): Section[] => {
    const matching = (assetLibrary.items ?? []).filter(
      (asset) => fits(asset.kind) && matchesWords(query, asset.name),
    )
    return ORDER.map((origin) => ({
      key: origin,
      label: LIBRARY_LABELS[origin],
      tiles: matching
        .filter((asset) => asset.origin === origin)
        .sort((a, b) => a.name.localeCompare(b.name))
        .map((asset) => ({
          key: asset.name,
          caption: leafName(asset.name),
          sub: asset.folder,
          thumb: hasThumbnail(asset)
            ? assetThumbnailUrl(asset.name)
            : undefined,
          glyph: GLYPHS[asset.kind] ?? '¶',
          current: asset.name === currentAsset,
          pick: () => reference(ASSET, asset.name),
        })),
    })).filter((section) => section.tiles.length)
  })

  // By workflow, the one run most recently first, and newest first within
  // (the listing is newest first, so first seen is newest)
  const outputSections = $derived.by((): Section[] => {
    const matching = (outputLibrary.items ?? []).filter(
      (file) => fits(file.kind) && matchesWords(query, file.name, file.label),
    )
    const folders = [...new Set(matching.map((file) => file.folder))]
    // A latest name marks the newest file it would resolve to
    const marked = currentOutput ? outputNamed(currentOutput)?.name : null
    return folders.map((folder) => ({
      key: folder,
      label: folder || 'outputs',
      tiles: matching
        .filter((file) => file.folder === folder)
        .map((file) => ({
          key: file.name,
          caption: leafName(file.name),
          sub: [file.version ? `v${file.version}` : '', file.subfolder]
            .filter(Boolean)
            .join(' · '),
          thumb: hasThumbnail(file)
            ? galleryThumbnailUrl(file.name)
            : undefined,
          glyph: GLYPHS[file.kind] ?? '¶',
          current: file.name === marked,
          pick: () =>
            reference(OUTPUT, (followLatest && latestName(file)) || file.name),
        })),
    }))
  })

  const sections = $derived(tab === 'assets' ? assetSections : outputSections)
  const listing = $derived(tab === 'assets' ? assetLibrary : outputLibrary)
  const what = $derived(tab === 'assets' ? 'the asset library' : 'earlier runs')

  function choose(tile: Tile) {
    onpick(tile.pick())
    open = false
    query = ''
  }
</script>

<Modal bind:open label="choose a file" large>
  <div class="head">
    <h2>Choose {kind === 'image' ? 'an image' : `a ${kind ?? 'file'}`}</h2>
    <div class="tabs" role="group" aria-label="where to choose from">
      <button
        class="quiet"
        class:on={tab === 'assets'}
        aria-pressed={tab === 'assets'}
        onclick={() => (tab = 'assets')}>Assets</button
      >
      <button
        class="quiet"
        class:on={tab === 'outputs'}
        aria-pressed={tab === 'outputs'}
        onclick={() => (tab = 'outputs')}>Earlier runs</button
      >
    </div>
    <input
      class="search"
      placeholder="search…"
      aria-label="search {what}"
      bind:value={query}
    />
  </div>
  {#if tab === 'outputs'}
    <label class="latest muted">
      <input type="checkbox" bind:checked={followLatest} />
      <span>
        Follow the newest run that has the file (<code>latest</code>), not the
        one picked
      </span>
    </label>
  {/if}
  <div class="body">
    {#if listing.items === undefined || listing.workspace === undefined}
      <p class="muted">Loading {what}…</p>
    {:else if sections.length === 0}
      <p class="muted">
        {query
          ? `Nothing in ${what} matches "${query}".`
          : `Nothing in ${what} yet.`}
      </p>
    {/if}
    {#each sections as section (section.key)}
      <h3 class="library">
        {section.label} <span class="muted">({section.tiles.length})</span>
      </h3>
      <div class="grid">
        {#each section.tiles as tile (tile.key)}
          <button
            class="tile"
            class:current={tile.current}
            aria-current={tile.current ? 'true' : undefined}
            onclick={() => choose(tile)}
            title={tile.key}
            aria-label="choose {tile.key}"
          >
            {#if tile.thumb}
              <img src={tile.thumb} alt="" loading="lazy" />
            {:else}
              <span class="glyph" aria-hidden="true">{tile.glyph}</span>
            {/if}
            <span class="caption">{tile.caption}</span>
            {#if tile.sub}<span class="folder muted">{tile.sub}</span>{/if}
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
  .tabs {
    display: flex;
    gap: var(--space-1);
  }
  .tabs .on {
    border-color: var(--accent);
    color: var(--ink);
  }
  .latest {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--t-sm);
    margin-bottom: var(--space-2);
  }
  .latest input {
    width: auto;
    flex: none;
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
