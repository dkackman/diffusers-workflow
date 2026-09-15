<script lang="ts">
  // The input side of the gallery (#165). Until this page existed there was
  // no way to see what the asset library held short of reading an `asset:`
  // failure or grepping the box, so writing a workflow that reuses an asset
  // meant guessing a name - and nothing showed which names a `common` or an
  // examples library was shadowing. The UX is the gallery's on purpose:
  // folder groups, a contact-sheet grid, a detail popout.
  import {
    ChevronDown,
    ChevronRight,
    FolderOpen,
    Trash2,
    Upload,
    X,
  } from '@lucide/svelte'
  import { api } from '../api'
  import CopyButton from '../CopyButton.svelte'
  import Empty from '../Empty.svelte'
  import FolderGroups from '../FolderGroups.svelte'
  import HintBar from '../HintBar.svelte'
  import { confirmDialog } from '../confirm.svelte'
  import BulkBar from '../BulkBar.svelte'
  import { Picks, actOnEach, dialogOpen } from '../picks.svelte'
  import { notify } from '../toast'
  import { storageGet, storageSet } from '../storage'
  import type { AssetFile, AssetLibrary, ShadowedAsset } from '../types'
  import WorkspacePicker from '../WorkspacePicker.svelte'
  import { workspace } from '../workspace.svelte'

  type Origin = AssetFile['origin']

  const COLLAPSE_KEY = 'collapsed-asset-libraries'
  // What each library is called in the page's own voice. `common` is the
  // shared library and `examples` a read-only tree an --examples-dir
  // brought; neither name means anything to someone who has not read the
  // server's flags
  const LIBRARY_LABELS: Record<Origin, string> = {
    workspace: 'This workspace',
    common: 'Shared library',
    examples: 'Examples',
  }
  // The same libraries in the possessive, for "shadowed by ...". Built by
  // hand rather than off the labels: "the examples's" is not English
  const SHADOWED_BY: Record<Origin, string> = {
    workspace: "this workspace's",
    common: "the shared library's",
    examples: "the examples library's",
  }

  let assets = $state<AssetFile[]>([])
  // The search path itself, in order, and the entries a nearer library on
  // it hides. Both come from the server: the page never builds a path
  let libraries = $state<AssetLibrary[]>([])
  let shadowed = $state<ShadowedAsset[]>([])
  let loaded = $state(false)
  let error = $state('')
  let filter = $state('')
  let selected = $state<AssetFile | null>(null)
  let busy = $state(false)
  // Which library the next upload lands in, set by whichever section's
  // button was clicked. The one thing about an upload that cannot be
  // changed afterwards, so it is the destination's own button rather than
  // a pick that can be left pointing at a library the page no longer shows
  let uploadTarget: 'workspace' | 'shared' = 'workspace'
  let fileInput = $state<HTMLInputElement | null>(null)
  // Which library sections are shut. Persisted, since a box whose shared
  // library dwarfs the workspace's own is exactly where one gets shut
  let collapsed = $state<Record<string, boolean>>(
    storageGet(COLLAPSE_KEY, {} as Record<string, boolean>),
  )

  function load() {
    return api
      .listAssets()
      .then((result) => {
        assets = result.assets
        libraries = result.libraries
        shadowed = result.shadowed
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

  const filterActive = $derived(filter !== '')
  const matches = (name: string) =>
    name.toLowerCase().includes(filter.toLowerCase())
  const visible = $derived(assets.filter((a) => matches(a.name)))
  const byName = $derived(new Map(visible.map((a) => [a.name, a])))
  const folderByName = $derived(new Map(assets.map((a) => [a.name, a.folder])))
  const originOf = $derived(new Map(assets.map((a) => [a.name, a.origin])))

  // One section per library on the search path, in the order the server
  // resolves them - the path is the page's top level and folders sit
  // inside it, because which library a name is in is what decides whether
  // it can be deleted, what a delete costs, and what it hides. An empty
  // library still shows its header, so an empty workspace says where an
  // upload would land; while a filter is on, a section with nothing to
  // show is skipped instead
  const sections = $derived(
    libraries
      .map((library) => ({
        ...library,
        label: LIBRARY_LABELS[library.origin] ?? library.origin,
        assets: visible.filter((a) => a.origin === library.origin),
        shadowed: shadowed.filter(
          (s) => s.origin === library.origin && matches(s.name),
        ),
      }))
      .filter(
        (section) =>
          !filterActive ||
          section.assets.length > 0 ||
          section.shadowed.length > 0,
      ),
  )
  // The grid in the order the page lays it out: what a shift-range spans
  // and the order the bulk actions act in
  const ordered = $derived(sections.flatMap((section) => section.assets))

  // Names ticked in the grid, for the bulk actions
  const picks = new Picks(() => ordered.map((a) => a.name))

  function toggleLibrary(origin: string) {
    collapsed[origin] = !collapsed[origin]
    storageSet(COLLAPSE_KEY, $state.snapshot(collapsed))
  }
  // While filtering, everything stays open - a shut section hiding matches
  // would make the filter look broken, as FolderGroups has it
  const isOpen = (origin: string) => filterActive || !collapsed[origin]

  function startUpload(target: 'workspace' | 'shared') {
    uploadTarget = target
    fileInput?.click()
  }

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
    // A shared asset goes for every workspace under the root, which is a
    // different thing from deleting one of this workspace's own - so the
    // confirm counts them rather than leaving the selection to be read
    const shared = names.filter(
      (name) => originOf.get(name) === 'common',
    ).length
    if (
      !(await confirmDialog(
        `Delete ${names.length} asset${names.length === 1 ? '' : 's'}?` +
          (shared
            ? ` ${shared} ${shared === 1 ? 'is' : 'are'} in the shared ` +
              `library and go${shared === 1 ? 'es' : ''} away for every ` +
              `workspace.`
            : '') +
          ` Any workflow still carrying one of those references stops loading.`,
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
      `Upload — name in the ${uploadTarget} asset library:`,
      suggestion,
    )
    if (assetName === null) return
    busy = true
    try {
      const result = await api.uploadMedia(
        file,
        assetName || undefined,
        uploadTarget === 'shared',
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
    const question =
      asset.origin === 'common'
        ? `Delete ${asset.reference} from the shared library? Every ` +
          `workspace under this root loses it, and any workflow still ` +
          `carrying that reference stops loading.`
        : `Delete ${asset.reference}? Any workflow still carrying that ` +
          `reference stops loading.`
    if (!(await confirmDialog(question, { confirmLabel: 'Delete' }))) return
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
  <span class="num muted">{assets.length} files</span>
  <input class="filter" placeholder="filter…" bind:value={filter} />
  <!-- One input for every section: which library the file lands in is
       uploadTarget, set by whichever button opened it -->
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
  loads this file at run time, whatever run produced it. The shared library and
  any examples library sit on every workspace's search path, so they follow you
  between workspaces; only this workspace's own section changes with the picker,
  and a name here hides the same name further down.
</HintBar>

{#if loaded && !error && assets.length === 0}
  <Empty>
    {#snippet icon()}<FolderOpen size={36} strokeWidth={1.5} />{/snippet}
    Nothing in the asset library yet — upload a file, or keep one from the gallery.
  </Empty>
{/if}

<BulkBar
  {picks}
  visible={ordered.length}
  {filterActive}
  {busy}
  noun="asset"
  ondownload={downloadPicked}
  ondelete={removePicked}
/>

{#each sections as section (section.origin)}
  <div class="libraryrow">
    <button
      class="library"
      onclick={() => toggleLibrary(section.origin)}
      title={isOpen(section.origin)
        ? 'collapse this library'
        : 'expand this library'}
    >
      {#if isOpen(section.origin)}<ChevronDown size={15} />{:else}<ChevronRight
          size={15}
        />{/if}
      {section.label}
      <span class="muted">({section.assets.length})</span>
    </button>
    <span class="path muted">{section.dir}</span>
    <span class="flex"></span>
    {#if !section.writable}
      <span class="muted" title="read-only: this server cannot write to it"
        >read-only</span
      >
    {:else if section.origin === 'common'}
      <button
        class="quiet withicon"
        onclick={() => startUpload('shared')}
        disabled={busy}
        title="lands in the shared library - visible from every workspace under this root and cannot be moved afterwards"
      >
        <Upload size={14} />Upload to shared
      </button>
    {:else if section.origin === 'workspace'}
      <button
        class="withicon"
        onclick={() => startUpload('workspace')}
        disabled={busy}
      >
        <Upload size={14} />Upload
      </button>
    {/if}
  </div>

  {#if isOpen(section.origin)}
    <FolderGroups
      names={section.assets.map((a) => a.name)}
      groupOf={(name) => folderByName.get(name) ?? ''}
      collapseKey="collapsed-asset-folders-{section.origin}"
      {filterActive}
      minColumn="150px"
    >
      {#snippet card(name)}
        {@const asset = byName.get(name)!}
        <div class="cellwrap" class:picked={picks.has(name)}>
          <!-- A sibling of the cell rather than a child: a checkbox nested in
               a button is invalid, and keeping them apart is what lets a plain
               click still open the details it always has -->
          {#if section.writable}
            <input
              class="pick"
              type="checkbox"
              checked={picks.has(name)}
              aria-label="select {asset.name}"
              title="select this asset for a bulk action"
              onclick={(e) => picks.toggle(name, e.shiftKey)}
            />
          {/if}
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
            <span class="caption" title={asset.reference}
              >{leaf(asset.name)}</span
            >
          </button>
        </div>
      {/snippet}
    </FolderGroups>

    {#if section.shadowed.length}
      <!-- What this library holds under a name a nearer one has taken. It
           is here because "I uploaded it and asset: still loads the old
           one" is otherwise unanswerable from the page - the server does
           not serve these, so a tile is a dimmed label rather than a
           picture, and nothing bulk can reach it -->
      <div class="grouprow">
        <span class="group"
          >shadowed/ <span class="muted">({section.shadowed.length})</span
          ></span
        >
      </div>
      <div class="grid">
        {#each section.shadowed as entry (entry.name)}
          <div class="cellwrap">
            <div
              class="cell shadowed"
              title="shadowed by {SHADOWED_BY[
                entry.shadowed_by
              ]} {entry.name} - {entry.reference} resolves to that file"
            >
              <span class="ghost">{entry.kind}</span>
              <span class="caption">{leaf(entry.name)}</span>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  {/if}
{/each}

{#if loaded && assets.length > 0 && ordered.length === 0}
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
  /* A library section header: the label, the count, the root it reads,
     and the one action that library offers */
  .libraryrow {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
    margin: 1.6rem 0 var(--space-2);
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--line);
  }
  /* A library is a heavier folder heading: the same mono name the engine
     resolves, ink rather than muted, because it is the page's top level */
  .library {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    background: none;
    border: none;
    color: var(--ink);
    font-family: var(--font-mono);
    font-weight: 600;
    font-size: var(--t-md);
    padding: 0;
    margin: 0;
    cursor: pointer;
  }
  .library:hover {
    filter: none;
    color: var(--accent);
  }
  /* FolderGroups' folder heading, for the one group it does not lay out.
     Its styles are scoped to that component, so this is the same look
     rather than the same rule */
  .grouprow {
    margin: 1.2rem 0 var(--space-2);
  }
  .group {
    color: var(--muted);
    font-family: var(--font-mono);
    font-weight: 600;
    font-size: var(--t-sm);
    letter-spacing: -0.01em;
  }
  .grid {
    display: grid;
    align-items: start;
    gap: 0.6rem;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
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
  /* The audio placeholder, and the shadowed tile's - neither has a
     picture to show, and both have to keep the grid's rhythm */
  .cell .audio,
  .cell .ghost {
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
  /* A shadowed entry is a name, not a file: the server serves the tile
     that won, so there is nothing to show and nothing to do with it */
  .cell.shadowed {
    opacity: 0.45;
    cursor: default;
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
  .path {
    font-family: var(--font-mono);
    word-break: break-all;
  }
</style>
