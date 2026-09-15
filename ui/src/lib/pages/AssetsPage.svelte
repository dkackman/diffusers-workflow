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
  import BulkBar from '../BulkBar.svelte'
  import { Picks, actOnEach, dialogOpen } from '../picks.svelte'
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
  // Names ticked in the grid, for the bulk actions. The grid order it is
  // given is what a shift-range spans and the order the actions act in
  const picks = new Picks(() => visible.map((a) => a.name))
  let busy = $state(false)
  // Where the next upload lands: this workspace's own library, or the
  // shared one every workspace under the root can see. The one thing about
  // an upload that cannot be changed afterwards, so it is named rather than
  // left as a bare 'shared' tickbox nothing on the page explains
  let uploadTo = $state<'workspace' | 'shared'>('workspace')
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
        picks.keepOnly(result.assets.map((a) => a.name))
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
  // How much of the grid a shared or examples library put there. Those
  // libraries sit on every workspace's search path, so this is the count
  // that stays the same when the workspace changes - unexplained, it reads
  // as a page that ignored the pick
  const borrowed = $derived(
    assets.filter((a) => a.origin !== 'workspace').length,
  )
  // Offered whenever anything came from somewhere else. A workspace with no
  // assets of its own is exactly when the question "why are these the same
  // as the last workspace?" comes up, and it was exactly when a
  // two-origins-or-more rule made the control disappear
  const originOffered = $derived(borrowed > 0)
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

  async function downloadPicked() {
    if (picks.names.length === 0) return
    busy = true
    try {
      await api.archiveAssets(picks.names)
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
    } finally {
      busy = false
    }
  }

  async function removePicked() {
    const names = picks.names
    if (names.length === 0) return
    if (
      !(await confirmDialog(
        `Delete ${names.length} asset${names.length === 1 ? '' : 's'}? Any ` +
          `workflow still carrying one of those references stops loading.`,
        { confirmLabel: 'Delete' },
      ))
    )
      return
    busy = true
    const failed = await actOnEach(names, (name) => api.deleteAsset(name))
    // A read-only examples asset answers 403; whatever could not go stays
    // ticked, so a retry needs no re-ticking
    picks.keepFailed(names, failed)
    if (failed.length)
      notify.error(
        `Could not delete ${failed.length} of ${names.length} assets: ${failed.join(', ')}`,
      )
    busy = false
    await load()
  }

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
      `Upload — name in the ${uploadTo === 'shared' ? 'shared' : 'workspace'} asset library:`,
      suggestion,
    )
    if (assetName === null) return
    busy = true
    try {
      const result = await api.uploadMedia(
        file,
        assetName || undefined,
        uploadTo === 'shared',
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
    // Escape closes the thing on top, and a dialog answers it itself. The
    // selection is the more recent, more surprising state to be stuck in,
    // so it clears first and the detail panel on a second press
    if (e.key !== 'Escape' || dialogOpen()) return
    if (picks.size) picks.clear()
    else selected = null
  }}
/>

<div class="head">
  <h1>Assets</h1>
  <WorkspacePicker />
  <!-- The borrowed clause is one expression rather than markup inside the
       block: Svelte trims the leading whitespace of a block's text, so a
       literal space before the separator does not survive a line wrap -->
  <span
    class="num muted"
    title={borrowed
      ? "from the shared or example libraries on every workspace's search path - a workspace change does not change these"
      : undefined}
    >{assets.length} files{#if borrowed}{` · ${borrowed} from other libraries`}{/if}</span
  >
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
  <label class="uploadto">
    upload to
    <select
      aria-label="where an upload lands"
      title="which library the next upload is written to - a shared one is
visible from every workspace and cannot be changed afterwards"
      bind:value={uploadTo}
    >
      <option value="workspace">this workspace</option>
      <option value="shared">shared library</option>
    </select>
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
  loads this file at run time, whatever run produced it. Assets from the shared
  and example libraries are on every workspace's search path, so they appear
  here whatever workspace you pick - only this workspace's own change with it,
  and a name here shadows the same name in a shared or example library.
</HintBar>

{#if loaded && !error && assets.length === 0}
  <Empty>
    {#snippet icon()}<FolderOpen size={36} strokeWidth={1.5} />{/snippet}
    Nothing in the asset library yet — upload a file, or keep one from the gallery.
  </Empty>
{/if}

<BulkBar
  {picks}
  visible={visible.length}
  {filterActive}
  {busy}
  noun="asset"
  ondownload={downloadPicked}
  ondelete={removePicked}
/>

<FolderGroups
  names={visible.map((a) => a.name)}
  groupOf={(name) => folderByName.get(name) ?? ''}
  collapseKey="collapsed-asset-folders"
  {filterActive}
  minColumn="150px"
>
  {#snippet card(name)}
    {@const asset = byName.get(name)!}
    <div class="cellwrap" class:picked={picks.has(name)}>
      <!-- A sibling of the cell rather than a child: a checkbox nested in
           a button is invalid, and keeping them apart is what lets a plain
           click still open the details it always has -->
      <input
        class="pick"
        type="checkbox"
        checked={picks.has(name)}
        aria-label="select {asset.name}"
        title="select this asset for a bulk action"
        onclick={(e) => picks.toggle(name, e.shiftKey)}
      />
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
  .originpick {
    /* The global select rule is width: 100% - here it must share the row */
    width: auto;
    max-width: 200px;
    font-family: var(--font-mono);
    font-size: var(--t-sm);
  }
  .uploadto {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    color: var(--muted);
    font-size: var(--t-xs);
  }
  .uploadto select {
    /* The global select rule is width: 100% - here it must share the row */
    width: auto;
    font-family: var(--font-mono);
    font-size: var(--t-sm);
  }
  .hiddenfile {
    display: none;
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
    /* Right, not left: the bulk-select checkbox owns the other corner */
    right: 0.3rem;
    padding: 0.05rem 0.35rem;
    border-radius: 999px;
    background: var(--panel);
    border: 1px solid var(--line);
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: 0.65rem;
  }
  /* The gallery's popout: it rides the bottom of the viewport while the
     grid scrolls behind it. Sitting at the end of the document instead
     would put it off screen for any click above the fold */
  .detail {
    position: sticky;
    bottom: 1rem;
    z-index: 2;
    margin-top: var(--space-4);
    padding: 0.6rem 0.8rem;
    box-shadow: 0 6px 24px rgb(0 0 0 / 0.35);
  }
  .bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
  }
  .flex {
    flex: 1;
  }
  .withicon {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
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
