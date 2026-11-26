<script lang="ts">
  // Browse for a file a field can name and pick one by eye: the asset
  // library, or what earlier runs wrote. Typing asset: or output: in a field
  // suits someone who knows the name; this suits picking by what it shows.
  import { untrack } from 'svelte'
  import Modal from './Modal.svelte'
  import Segmented from './Segmented.svelte'
  import GroupHeading from './GroupHeading.svelte'
  import { collapsedState } from '../collapsed.svelte'
  import OutputTile from './OutputTile.svelte'
  import { assetMedia, outputMedia, type MediaFile } from '../media'
  import { assetLibrary, ensureAssetLibrary } from '../assetlib.svelte'
  import {
    ensureOutputLibrary,
    latestName,
    loadOutputLibrary,
    outputLibrary,
    outputNamed,
  } from '../outputlib.svelte'
  import { ASSET, OUTPUT, reference, referenceName } from '../references'
  import { kindWithArticle } from '../referenceField'
  import { matchesWords } from './suggest'
  import { compareNames } from '../names'
  import type { AssetFile } from '../types'
  import { workspace } from '../workspace.svelte'

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
    file: MediaFile
    sub: string
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
        .sort((a, b) => compareNames(a.name, b.name))
        .map((asset) => ({
          key: asset.name,
          file: assetMedia(asset, assetLibrary.workspace ?? workspace.current),
          sub: asset.folder,
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
          file: outputMedia(file, outputLibrary.workspace ?? workspace.current),
          sub: [file.version ? `v${file.version}` : '', file.subfolder]
            .filter(Boolean)
            .join(' · '),
          current: file.name === marked,
          pick: () =>
            reference(OUTPUT, (followLatest && latestName(file)) || file.name),
        })),
    }))
  })

  const sections = $derived(tab === 'assets' ? assetSections : outputSections)
  const listing = $derived(tab === 'assets' ? assetLibrary : outputLibrary)
  // A section the person shut stays shut next time, except while a search is
  // on: a shut section hiding matches would make the search look broken
  // and per workspace and tab, since a library's names are its own
  const shut = collapsedState(
    () => `collapsed-media-picker:${workspace.current}:${tab}`,
  )
  const closed = (key: string) => !query && shut.has(key)
  const searching = 'Showing every match while the search is on'
  const what = $derived(tab === 'assets' ? 'the asset library' : 'earlier jobs')

  function choose(tile: Tile) {
    onpick(tile.pick())
    open = false
    query = ''
  }
</script>

<Modal bind:open label="choose a file" size="large">
  <div class="head">
    <h2>Choose {kindWithArticle(kind)}</h2>
    <Segmented
      label="where to choose from"
      value={tab}
      options={[
        { value: 'assets', label: 'Assets' },
        { value: 'outputs', label: 'Earlier jobs' },
      ]}
      onselect={(picked) => (tab = picked)}
    />
    <input
      class="search"
      placeholder="search…"
      aria-label="search {what}"
      bind:value={query}
    />
  </div>
  {#if tab === 'outputs'}
    <label class="latest">
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
        <GroupHeading
          label={section.label}
          count={section.tiles.length}
          collapsed={closed(section.key)}
          ontoggle={() => shut.toggle(section.key)}
          disabled={!!query}
          title={query ? searching : undefined}
        />
      </h3>
      {#if !closed(section.key)}
        <div class="grid">
          {#each section.tiles as tile (tile.key)}
            {#snippet folder()}<span class="folder">{tile.sub}</span>{/snippet}
            {#snippet here()}<span class="tile-mark here">
                current
              </span>{/snippet}
            <OutputTile
              file={tile.file}
              size="strip"
              alt=""
              active={tile.current}
              aria-current={tile.current ? 'true' : undefined}
              onclick={() => choose(tile)}
              title={tile.key}
              aria-label="choose {tile.key}"
              badges={tile.current ? here : undefined}
              meta={tile.sub ? folder : undefined}
            />
          {/each}
        </div>
      {/if}
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
    grid-template-columns: repeat(auto-fill, 96px);
    gap: var(--space-2);
  }
  /* Where it is in its library: the folder, or which run */
  .folder {
    display: block;
    font-family: var(--font-mono);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
