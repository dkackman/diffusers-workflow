<script lang="ts">
  // The input side of the gallery (#165). Until this page existed there was
  // no way to see what the asset library held short of reading an `asset:`
  // failure or grepping the box, so writing a workflow that reuses an asset
  // meant guessing a name - and nothing showed which names a `common` or an
  // examples library was shadowing. The UX is the gallery's on purpose:
  // folder groups, a contact-sheet grid, a detail popout.
  import { FolderOpen } from '@lucide/svelte'
  import { api } from '../api'
  import { assetMedia } from '../media'
  import OutputTile from '../ui/OutputTile.svelte'
  import { untrack } from 'svelte'
  import { resource } from '../resource.svelte'
  import { applyEdits, noEdits, settleEdits, withPatch } from '../gallery/order'
  import LoadState from '../ui/LoadState.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import FolderGroups from '../FolderGroups.svelte'
  import { dropFiles } from '../dropFiles'
  import HintBar from '../HintBar.svelte'
  import { action } from '../action.svelte'
  import { assetConfirm, assetsConfirm } from '../assets/deletion'
  import BulkBar from '../BulkBar.svelte'
  import { Picks, actOnEach, reportFailures } from '../picks.svelte'
  import { overlayOpen } from '../ui/layers.svelte'
  import { notify } from '../toast'
  import { collapsedState } from '../collapsed.svelte'
  import type { AssetFile, AssetLibrary, ShadowedAsset } from '../types'
  import { workspace } from '../workspace.svelte'
  import { sharedHref } from '../routes'
  import { hashState } from '../router.svelte'
  import { assetLibrary } from '../assetlib.svelte'
  import {
    UPLOAD_EXTENSIONS,
    reportUploads,
    uploadAll,
  } from '../assets/uploads'
  import AssetDetail from '../assets/AssetDetail.svelte'
  import LibraryRow from '../assets/LibraryRow.svelte'
  import ShadowedAssets from '../assets/ShadowedAssets.svelte'

  type Origin = AssetFile['origin']

  let { shared = false }: { shared?: boolean } = $props()

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

  // An asset an agent adds or removes shows within one feed interval. The
  // search path itself, in order, and the entries a nearer library on it
  // hides, come from the server: the page never builds a path
  const list = resource(
    async () => {
      // The note saves this read can have seen: those made before it began
      const began = untrack(() => noteEdits.made)
      return { ...(await api.listAssets()), began }
    },
    {
      watch: ['assets'],
      rows: (data) => data.assets.map((asset) => [asset.name, asset]),
      // What is held carries the notes saved here still open, so a quiet
      // reload never marks the page's own note (the gallery's merge)
      merge: (_held, incoming) => {
        noteEdits = settleEdits(incoming.assets, incoming.began, noteEdits)
        return { ...incoming, assets: applyEdits(incoming.assets, noteEdits) }
      },
    },
  )
  // A note saved here, until a listing begun after the save lands (the
  // gallery's edits, gallery/order.ts)
  let noteEdits = $state(noEdits<AssetFile>())
  const assets = $derived(applyEdits(list.data?.assets ?? [], noteEdits))
  const libraries = $derived<AssetLibrary[]>(list.data?.libraries ?? [])
  const shadowed = $derived<ShadowedAsset[]>(list.data?.shadowed ?? [])
  // In the hash, so Back from an asset - or a shared link - keeps it
  const filter = hashState('q', '')
  let selected = $state<AssetFile | null>(null)
  // Which library the next upload lands in, set by whichever section's
  // button was clicked. The one thing about an upload that cannot be
  // changed afterwards, so it is the destination's own button rather than
  // a pick that can be left pointing at a library the page no longer shows
  let uploadTarget: 'workspace' | 'shared' = 'workspace'
  let fileInput = $state<HTMLInputElement | null>(null)
  // Which library sections are shut. Persisted, since a box whose shared
  // library dwarfs the workspace's own is exactly where one gets shut
  const collapsed = collapsedState(COLLAPSE_KEY)

  // Carry a saved note onto the listing and the pick, so selecting the asset
  // again shows it rather than what was fetched before the edit
  function saveNote(note: string | null) {
    if (!selected) return
    const name = selected.name
    noteEdits = withPatch(noteEdits, name, { note: note ?? undefined })
    selected = { ...selected, note: note ?? undefined }
    // Held as read (resource's `adopt`): never marked as another writer's
    const held = list.data
    if (held)
      list.adopt({ ...held, assets: applyEdits(held.assets, noteEdits) })
  }

  const filterActive = $derived(filter.value !== '')
  const matches = (name: string) =>
    name.toLowerCase().includes(filter.value.toLowerCase())
  const visible = $derived(assets.filter((a) => matches(a.name)))
  const byName = $derived(new Map(visible.map((a) => [a.name, a])))
  const folderByName = $derived(new Map(assets.map((a) => [a.name, a.folder])))
  const originOf = $derived(new Map(assets.map((a) => [a.name, a.origin])))
  // A workspace's page shows the shared and examples libraries only as far
  // as its own workflows reach into them - the whole shared library is the
  // Shared assets page's, and listing it in every workspace buried what the
  // workspace made. Those workflows resolve their `asset:` names through
  // the workspace first, so a hidden shared copy is shown when the name is
  // one they use
  const onPage = (asset: AssetFile) =>
    shared
      ? asset.origin !== 'workspace'
      : asset.origin === 'workspace' || !!asset.used_by?.length
  const shownCount = $derived(assets.filter(onPage).length)
  const usedNames = $derived(
    new Set(assets.filter((a) => a.used_by?.length).map((a) => a.name)),
  )
  const elsewhereCount = $derived(
    assets.filter((a) => a.origin !== 'workspace').length,
  )

  // While filtering, everything stays open - a shut section hiding matches
  // would make the filter look broken, as FolderGroups has it
  const isOpen = (origin: string) => filterActive || !collapsed.has(origin)

  // One section per library on the search path, in the order the server
  // resolves them - the path is the page's top level and folders sit
  // inside it, because which library a name is in is what decides whether
  // it can be deleted, what a delete costs, and what it hides. An empty
  // library still shows its header, so an empty workspace says where an
  // upload would land; while a filter is on, a section with nothing to
  // show is skipped instead
  const shows = (section: { assets: unknown[]; shadowed: unknown[] }) =>
    section.assets.length > 0 || section.shadowed.length > 0
  const sections = $derived(
    libraries
      .map((library) => ({
        ...library,
        label:
          (LIBRARY_LABELS[library.origin] ?? library.origin) +
          (shared || library.origin === 'workspace' ? '' : ', used here'),
        assets: visible.filter((a) => a.origin === library.origin && onPage(a)),
        shadowed: shadowed.filter(
          (s) =>
            s.origin === library.origin &&
            matches(s.name) &&
            (shared || usedNames.has(s.name)),
        ),
      }))
      .filter((section) => !shared || section.origin !== 'workspace')
      .filter(
        (section) =>
          section.assets.length > 0 ||
          section.shadowed.length > 0 ||
          // Unfiltered, an empty library still shows its header, so an
          // empty workspace says where an upload would land - but on a
          // workspace's page a shared library it uses nothing from is left
          // out rather than shown as an empty "used here"
          (!filterActive && (shared || section.origin === 'workspace')),
      ),
  )
  // What the bulk actions can reach: the grid in the order the page lays
  // it out, minus anything not on screen or not tickable. A shut section
  // is hidden, and a read-only library's tile carries no checkbox and
  // would answer 403 - selecting either is a bulk action touching what
  // the user cannot see or undo from the tile
  const ordered = $derived(
    sections
      .filter((section) => isOpen(section.origin) && section.writable)
      .flatMap((section) => section.assets),
  )

  // Names ticked in the grid, for the bulk actions
  const picks = new Picks(() => ordered.map((a) => a.name))

  // What a landed listing means to the page. No listing is a new workspace
  // (or the first read): nothing ticked. The editors' asset: completions
  // read the same listing; an upload or delete here is what they offer next
  $effect(() => {
    const data = list.data
    untrack(() => {
      if (!data) {
        // Another workspace's names; the count goes on, as reads in flight
        // carry it
        noteEdits = { ...noEdits(), made: noteEdits.made }
        return picks.clear()
      }
      assetLibrary.items = data.assets
      assetLibrary.workspace = workspace.current
      if (selected && !data.assets.some((a) => a.name === selected?.name))
        selected = null
    })
  })

  function startUpload(target: 'workspace' | 'shared') {
    uploadTarget = target
    fileInput?.click()
  }

  const downloading = picks.download(api.archiveAssets, 'assets')

  const removingPicked = action(
    async (names: string[]) => {
      const failed = await actOnEach(names, (name) => api.deleteAsset(name))
      // A read-only examples asset answers 403; whatever could not go stays
      // ticked, so a retry needs no re-ticking
      picks.keepFailed(
        names,
        failed.map((f) => f.name),
      )
      reportFailures('delete', failed, names.length, 'assets')
      await list.reload()
    },
    {
      doing: 'delete the picked assets',
      confirm: (names) =>
        assetsConfirm(
          names.length,
          names.filter((name) => originOf.get(name) === 'common').length,
        ),
    },
  )
  const removePicked = () =>
    picks.names.length && removingPicked.run(picks.names)

  function upload(event: Event) {
    const input = event.target as HTMLInputElement
    const files = Array.from(input.files ?? [])
    // Clear it either way, so picking the same file twice still fires
    input.value = ''
    uploadFiles(files, uploadTarget)
  }

  // Each file's own failure is reportUploads'; the action reports the rest
  const uploading = action(
    async (files: File[], target: 'workspace' | 'shared') => {
      const isShared = target === 'shared'
      // The names the target library already holds, so Keep both starts
      // from a free one
      const taken = new Set(
        assets
          .filter((a) => (a.origin === 'workspace') !== isShared)
          .map((a) => a.name),
      )
      reportUploads(await uploadAll(files, { shared: isShared, taken }))
      await list.reload()
    },
    { doing: 'upload the files' },
  )
  function uploadFiles(files: File[], target: 'workspace' | 'shared') {
    if (files.length) void uploading.run(files, target)
  }

  const removing = action(
    async (asset: AssetFile) => {
      await api.deleteAsset(asset.name)
      notify.success(`Deleted ${asset.reference}`)
      if (selected?.name === asset.name) selected = null
      await list.reload()
    },
    { doing: (asset) => `delete ${asset.name}`, confirm: assetConfirm },
  )
  const remove = (asset: AssetFile) => removing.run(asset)
  const working = $derived(
    uploading.busy || downloading.busy || removing.busy || removingPicked.busy,
  )
</script>

<svelte:window
  onkeydown={(e) => {
    // Escape closes the thing on top, and a dialog answers it itself. The
    // selection is the more recent, more surprising state to be stuck in,
    // so it clears first and the detail panel on a second press
    if (e.key !== 'Escape' || overlayOpen()) return
    if (picks.size) picks.clear()
    else selected = null
  }}
/>

<div
  class="dropzone"
  use:dropFiles={{
    ondrop: (files) => uploadFiles(files, shared ? 'shared' : 'workspace'),
    disabled: working,
  }}
>
  {#snippet search()}
    <input class="filter" placeholder="filter…" bind:value={filter.value} />
  {/snippet}
  <PageHeader
    title={shared ? 'Shared assets' : 'Assets'}
    count={{ n: shownCount, noun: 'file' }}
    filter={search}
  >
    {#snippet actions()}
      {#if !shared && elsewhereCount}
        <a
          class="sidelink plain muted"
          href={sharedHref('assets')}
          title="the shared library every workspace under this root can use"
        >
          shared <span class="num">{elsewhereCount}</span>
        </a>
      {/if}
      <!-- One input for every section: which library the file lands in is
       uploadTarget, set by whichever button opened it -->
      <input
        class="hiddenfile"
        type="file"
        multiple
        accept={UPLOAD_EXTENSIONS.join(',')}
        bind:this={fileInput}
        onchange={upload}
      />
    {/snippet}
  </PageHeader>

  <HintBar storageKey="assets-hint-dismissed">
    An asset is input a workflow names by reference: an argument set to
    asset:name loads this file at run time, whatever run produced it. Drop files
    anywhere on the page to upload them here.
    {#if shared}
      The shared library follows you between workspaces; an examples library is
      read-only.
    {:else}
      This workspace's own assets come first, and a name here hides the same
      name in the shared library. Shared assets are listed here only when one of
      this workspace's workflows names them; the rest are on the Shared assets
      page.
    {/if}
  </HintBar>

  <LoadState
    of={list}
    label="the asset library"
    empty={shownCount === 0 || (filterActive && !sections.some(shows))}
    filtered={filterActive && shownCount > 0}
    onclear={() => (filter.value = '')}
  >
    {#snippet emptyIcon()}<FolderOpen size={36} strokeWidth={1.5} />{/snippet}
    {#snippet emptyText()}
      Nothing in the asset library yet — upload a file, or keep one from the
      gallery.
      {#if !shared && elsewhereCount}
        The <a href={sharedHref('assets')}>shared library</a> has {elsewhereCount}
        more.
      {/if}
    {/snippet}
    <!-- No children: the grid is the library sections below, which need
         the empty library's headers too, so they sit outside this -->
  </LoadState>

  <BulkBar
    {picks}
    visible={ordered.length}
    {filterActive}
    busy={working}
    noun="asset"
    ondownload={downloading.run}
    ondelete={removePicked}
  />

  {#each sections as section (section.origin)}
    <LibraryRow
      {section}
      open={isOpen(section.origin)}
      {shared}
      busy={working}
      ontoggle={() => collapsed.toggle(section.origin)}
      onupload={startUpload}
    />

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
          <div
            class="cellwrap"
            class:picked={picks.has(name)}
            class:row-changed={list.changed.has(name)}
          >
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
            <OutputTile
              file={assetMedia(
                asset,
                list.data?.workspace ?? workspace.current,
              )}
              active={selected?.name === name}
              picked={picks.has(name)}
              onclick={() => (selected = asset)}
              aria-label="show details for {asset.name}"
              title="show details"
            />
          </div>
        {/snippet}
      </FolderGroups>

      {#if section.shadowed.length}
        <ShadowedAssets entries={section.shadowed} {filterActive} />
      {/if}
    {/if}
  {/each}
</div>

{#if selected}
  <!-- Keyed by the pick, so each asset's panel starts its own read and never
       shows the last one's -->
  {#key selected.name}
    <AssetDetail
      asset={selected}
      busy={working}
      onremove={remove}
      onclose={() => (selected = null)}
      onnotesaved={saveNote}
    />
  {/key}
{/if}

<style>
  .dropzone {
    min-height: 60vh;
  }
  .hiddenfile {
    display: none;
  }
</style>
