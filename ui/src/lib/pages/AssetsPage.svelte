<script lang="ts">
  // The input side of the gallery (#165). Until this page existed there was
  // no way to see what the asset library held short of reading an `asset:`
  // failure or grepping the box, so writing a workflow that reuses an asset
  // meant guessing a name - and nothing showed which names a `common` or an
  // examples library was shadowing. The UX is the gallery's on purpose:
  // folder groups, a contact-sheet grid, a detail popout.
  import { FolderOpen, Trash2, Upload, X } from '@lucide/svelte'
  import { api } from '../api'
  import CopyButton from '../CopyButton.svelte'
  import Empty from '../Empty.svelte'
  import FolderGroups from '../FolderGroups.svelte'
  import HintBar from '../HintBar.svelte'
  import { confirmDialog } from '../confirm.svelte'
  import { notify } from '../toast'
  import type { AssetFile } from '../types'
  import WorkspacePicker from '../WorkspacePicker.svelte'
  import { workspace } from '../workspace.svelte'

  let assets = $state<AssetFile[]>([])
  let assetDir = $state<string | null>(null)
  let assetDirs = $state<string[]>([])
  let loaded = $state(false)
  let error = $state('')
  let filter = $state('')
  // Which library to show - null is every one. Only offered once more than
  // one is on the search path, since a server with only a workspace library
  // has nothing to choose between
  let origin = $state<AssetFile['origin'] | null>(null)
  let selected = $state<AssetFile | null>(null)
  let busy = $state(false)
  // Put the next upload in the shared library rather than this workspace's,
  // which is the one thing about an upload that cannot be changed afterwards
  let uploadShared = $state(false)
  let fileInput = $state<HTMLInputElement | null>(null)

  function load() {
    return api
      .listAssets()
      .then((result) => {
        assets = result.assets
        assetDir = result.asset_dir
        assetDirs = result.asset_dirs
        error = ''
        if (selected && !result.assets.some((a) => a.name === selected?.name))
          selected = null
      })
      .catch((e) => (error = e.message))
      .finally(() => (loaded = true))
  }

  $effect(() => {
    // Read inside the effect so switching workspaces refetches the library
    void workspace.current
    void load()
  })

  const origins = $derived([...new Set(assets.map((a) => a.origin))].sort())
  const originOffered = $derived(origins.length > 1)
  const filterActive = $derived(filter !== '' || origin !== null)
  const visible = $derived(
    assets.filter(
      (a) =>
        a.name.toLowerCase().includes(filter.toLowerCase()) &&
        (origin === null || a.origin === origin),
    ),
  )
  const byName = $derived(new Map(visible.map((a) => [a.name, a])))
  const folderByName = $derived(new Map(assets.map((a) => [a.name, a.folder])))

  // A pick of a library that no longer has anything in it means every
  // library, not an empty grid pinned to a vanished value
  $effect(() => {
    if (origin !== null && !origins.includes(origin)) origin = null
  })

  const day = (mtime: number) => new Date(mtime * 1000).toLocaleString()
  const kb = (size: number) =>
    size < 1024 * 1024
      ? (size / 1024).toFixed(0) + ' KB'
      : (size / (1024 * 1024)).toFixed(1) + ' MB'
  const leaf = (name: string) => name.split('/').pop() ?? name

  async function upload(event: Event) {
    const input = event.target as HTMLInputElement
    const file = input.files?.[0]
    // Clear it either way, so picking the same file twice still fires
    input.value = ''
    if (!file) return
    const suggestion = file.name
    const assetName = window.prompt(
      `Upload — name in the ${uploadShared ? 'shared' : 'workspace'} asset library:`,
      suggestion,
    )
    if (assetName === null) return
    busy = true
    try {
      const result = await api.uploadMedia(
        file,
        assetName || undefined,
        uploadShared,
      )
      notify.success(`Uploaded as ${result.reference ?? result.path}`)
      await load()
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
    } finally {
      busy = false
    }
  }

  async function remove(asset: AssetFile) {
    if (
      !(await confirmDialog(
        `Delete ${asset.reference}? Any workflow still carrying that ` +
          `reference stops loading.`,
        { confirmLabel: 'Delete' },
      ))
    )
      return
    busy = true
    try {
      await api.deleteAsset(asset.name)
      notify.success(`Deleted ${asset.reference}`)
      if (selected?.name === asset.name) selected = null
      await load()
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
    } finally {
      busy = false
    }
  }
</script>

<svelte:window
  onkeydown={(e) => {
    if (e.key === 'Escape' && selected) selected = null
  }}
/>

<div class="head">
  <h1>Assets</h1>
  <WorkspacePicker />
  <span class="num muted">{assets.length} files</span>
  <input class="filter" placeholder="filter…" bind:value={filter} />
  {#if originOffered}
    <select
      class="originpick"
      aria-label="library"
      title="show only assets from one library"
      bind:value={origin}
    >
      <option value={null}>all libraries</option>
      {#each origins as o (o)}
        <option value={o}>{o}</option>
      {/each}
    </select>
  {/if}
  <label
    class="sharedtoggle"
    title="put the next upload in the library every workspace shares"
  >
    <input type="checkbox" bind:checked={uploadShared} />
    shared
  </label>
  <button class="withicon" onclick={() => fileInput?.click()} disabled={busy}>
    <Upload size={14} />Upload
  </button>
  <input
    class="hiddenfile"
    type="file"
    accept="image/*,video/*,audio/*"
    bind:this={fileInput}
    onchange={upload}
  />
</div>

{#if error}<p class="muted">Could not read the asset library: {error}</p>{/if}

<HintBar storageKey="assets-hint-dismissed">
  An asset is input a workflow names by reference: an argument set to asset:name
  loads this file at run time, whatever run produced it. A name in this
  workspace shadows the same name in a shared or example library.
</HintBar>

{#if loaded && !error && assets.length === 0}
  <Empty>
    {#snippet icon()}<FolderOpen size={36} strokeWidth={1.5} />{/snippet}
    Nothing in the asset library yet — upload a file, or keep one from the gallery.
  </Empty>
{/if}

<FolderGroups
  names={visible.map((a) => a.name)}
  groupOf={(name) => folderByName.get(name) ?? ''}
  collapseKey="collapsed-asset-folders"
  {filterActive}
  minColumn="150px"
>
  {#snippet card(name)}
    {@const asset = byName.get(name)!}
    <div class="cellwrap">
      <button
        class="cell"
        class:active={selected?.name === name}
        onclick={() => (selected = asset)}
        aria-label="show details for {asset.name}"
        title="show details"
      >
        {#if asset.kind === 'image'}
          <img src={asset.url} alt={asset.name} loading="lazy" />
        {:else if asset.kind === 'video'}
          <video src={asset.url} preload="metadata" muted></video>
        {:else}
          <span class="audio">♪ {leaf(asset.name)}</span>
        {/if}
        <span class="caption" title={asset.reference}>{leaf(asset.name)}</span>
      </button>
      {#if asset.origin !== 'workspace'}
        <span
          class="origin"
          title="from the {asset.origin} library - this workspace's own names shadow it"
          >{asset.origin}</span
        >
      {/if}
    </div>
  {/snippet}
</FolderGroups>

{#if loaded && assets.length > 0 && visible.length === 0}
  <p class="muted">Nothing matches "{filter}".</p>
{/if}

{#if selected}
  <div class="detail panel">
    <div class="bar">
      <strong class="selname">{selected.reference}</strong>
      <CopyButton
        text={selected.reference}
        title="copy the reference a workflow argument carries"
      />
      <span class="flex"></span>
      <a
        href={selected.url}
        target="_blank"
        class="muted"
        title="open the file itself in a new tab">open file</a
      >
      <span class="num muted"
        >{selected.kind} · {kb(selected.size)} · {day(selected.mtime)}</span
      >
      {#if selected.origin === 'examples'}
        <span class="muted" title="read-only: an examples library brought it"
          >read-only</span
        >
      {:else}
        <button
          class="quiet icon danger"
          onclick={() => selected && remove(selected)}
          disabled={busy}
          title="delete this asset from the library"
          aria-label="delete this asset from the library"
        >
          <Trash2 size={14} />
        </button>
      {/if}
      <span class="flex"></span>
      <button
        class="quiet icon"
        onclick={() => (selected = null)}
        title="close details"
        aria-label="close details"><X size={14} /></button
      >
    </div>
    <div class="body">
      {#if selected.kind === 'image'}
        <img src={selected.url} alt={selected.name} />
      {:else if selected.kind === 'video'}
        <!-- svelte-ignore a11y_media_has_caption -->
        <video src={selected.url} controls loop></video>
      {:else}
        <audio src={selected.url} controls></audio>
      {/if}
    </div>
  </div>
{/if}

{#if assetDirs.length}
  <p class="dir muted">
    read from
    {#each assetDirs as dir, index (dir)}
      <span class="path">{dir}</span>{#if index < assetDirs.length - 1},
      {/if}
    {/each}
    {#if assetDir}
      · uploads land in <span class="path">{assetDir}</span>
    {/if}
  </p>
{/if}

<style>
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.8rem;
    margin-bottom: var(--space-4);
  }
  .head .filter {
    max-width: 220px;
    margin-left: auto;
  }
  .sharedtoggle {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    color: var(--muted);
    font-size: var(--t-xs);
  }
  .hiddenfile {
    display: none;
  }
  .cellwrap {
    position: relative;
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
  }
  .cell:hover {
    border-color: var(--accent);
    filter: none;
  }
  .cell.active {
    border-color: var(--accent);
    box-shadow: inset 0 0 0 1px var(--accent);
  }
  .cell img,
  .cell video {
    display: block;
    width: 100%;
    height: 110px;
    object-fit: cover;
    background: var(--sunk);
  }
  .cell .audio {
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
  /* Which library an entry came from, on the tile rather than only in the
     detail - "why can I not delete this" has to be answerable at a glance */
  .origin {
    position: absolute;
    top: 0.3rem;
    left: 0.3rem;
    padding: 0.05rem 0.35rem;
    border-radius: 999px;
    background: var(--panel);
    border: 1px solid var(--line);
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: 0.65rem;
  }
  .detail {
    margin-top: var(--space-4);
    padding: 0.6rem 0.8rem;
  }
  .bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
  }
  .bar .flex {
    flex: 1;
  }
  .selname {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
    word-break: break-all;
  }
  .body {
    margin-top: 0.6rem;
  }
  .body img,
  .body video {
    max-width: 100%;
    max-height: 60vh;
    border-radius: var(--radius-frame);
    background: var(--sunk);
  }
  .body audio {
    width: 100%;
  }
  .dir {
    margin-top: 2.5rem;
    font-size: var(--t-xs);
  }
  .path {
    font-family: var(--font-mono);
    word-break: break-all;
  }
</style>
