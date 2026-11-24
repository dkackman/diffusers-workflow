<script lang="ts">
  import {
    Bookmark,
    Gauge,
    ImagePlus,
    ImageOff,
    Star,
    Trash2,
    X,
  } from '@lucide/svelte'
  import { outputLibrary } from '../outputlib.svelte'
  import { api } from '../api'
  import DownloadLink from '../DownloadLink.svelte'
  import { outputMedia, type MediaFile } from '../media'
  import { openInHash } from '../mediaViewer'
  import { leafName } from '../names'
  import { OUTPUT } from '../references'
  import OutputTile from '../ui/OutputTile.svelte'
  import MediaViewer from '../ui/MediaViewer.svelte'
  import RecipePanel from '../ui/RecipePanel.svelte'
  import ReferenceChip from '../ui/ReferenceChip.svelte'
  import { untrack } from 'svelte'
  import { resource, type Resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import Segmented from '../ui/Segmented.svelte'
  import FolderGroups from '../FolderGroups.svelte'
  import BulkBar from '../BulkBar.svelte'
  import { Picks, actOnEach, reportFailures } from '../picks.svelte'
  import { overlayOpen } from '../ui/layers.svelte'
  import { action } from '../action.svelte'
  import { askDeletion, fileConfirm } from '../gallery/deletion'
  import type { GalleryFile, OutputRole } from '../types'
  import { workspace } from '../workspace.svelte'
  import { codec, go, hashState } from '../router.svelte'
  import RequestFilter from '../gallery/RequestFilter.svelte'
  import RoleChips from '../gallery/RoleChips.svelte'
  import GalleryAside from '../gallery/GalleryAside.svelte'
  import { saveAsAsset } from '../gallery/saveAsAsset'
  import UseAsInput from '../gallery/UseAsInput.svelte'
  import { AssessCache } from '../gallery/assess.svelte'
  import RoleTag from '../gallery/RoleTag.svelte'
  import { versionTitle } from '../terms'
  import {
    GROUPINGS,
    GROUP_OPTIONS,
    SORTS,
    SORT_OPTIONS,
    applyEdits,
    gridFiles,
    gridGroups,
    inGroupOrder,
    keepPicks,
    listingRows,
    noEdits,
    settleEdits,
    withGone,
    withPatch,
    type GallerySort,
    type Grouping,
    type Listing,
  } from '../gallery/order'

  // The filters live in the hash, so opening a file and pressing Back - or
  // sharing the link - keeps the narrowed view
  const text = hashState('q', '')
  // The roles to show. Finals are what a run was for, so the gallery opens
  // on them; Intermediate adds the steps' working files
  const roles = hashState<OutputRole[]>(
    'roles',
    ['final'],
    codec.someOf<OutputRole>(['final', 'intermediate'], ['final']),
  )
  const sort = hashState('sort', SORTS[0], codec.oneOf(SORTS))
  const groupBy = hashState('group', GROUPINGS[0], codec.oneOf(GROUPINGS))
  // The file open in the viewer, as `open=` in the hash: the viewer writes
  // it the way Back expects (mediaViewer.ts), and Back closes it
  let open = $state<string | null>(openInHash())
  // The file Use as input… was chosen on, and its dialog
  let useInput = $state<MediaFile | null>(null)
  let useInputOpen = $state(false)
  // Names ticked in the grid, for the bulk actions. The grid order it is
  // given is what a shift-range spans and the order the actions act in
  const picks: Picks = new Picks(() => shown.map((f) => f.name))
  const assessCache = new AssessCache()

  // The requests-ledger request the gallery is narrowed to (RequestFilter
  // keeps it in step with the hash); the listing asks the server for it
  let reqId = $state('')
  let reqItem = $state('')

  // An output landing from outside shows within one feed interval. Only the
  // listing is read again: the open file and its note box are left as they are
  const list: Resource<Listing> = resource(
    async (): Promise<Listing> => {
      const key = `${reqId}\u0000${reqItem}`
      // The edits this read can have seen: those made before it began
      const began = untrack(() => edits.made)
      const result = await api.gallery(
        undefined,
        reqId || undefined,
        reqItem || undefined,
        { prompts: true },
      )
      const { files, prompts = {} } = result
      return { files, key, began, workspace: result.workspace, prompts }
    },
    {
      watch: ['gallery'],
      rows: listingRows,
      // What is held carries the page's edits still open, so a quiet reload
      // diffs against them and never marks the page's own star or note
      merge: (held, incoming) => {
        edits = settleEdits(incoming.files, incoming.began, edits)
        const files = applyEdits(incoming.files, edits)
        return keepPicks(picks, held, { ...incoming, files })
      },
    },
  )
  // What this page changed (a star, a delete, a note) rides on the listing
  // until a read begun after the change lands, so one asked for before it
  // cannot undo it
  let edits = $state(noEdits())
  // Each clip's and sound's length: the listing again with `media`, read in
  // the background so the grid never waits on the probe; a tile takes its
  // duration from here by name
  const durations = resource(
    async () => {
      const result = await api.gallery(
        undefined,
        reqId || undefined,
        reqItem || undefined,
        { media: true },
      )
      return new Map(result.files.map((f) => [f.name, f.duration_seconds]))
    },
    { watch: ['gallery'] },
  )
  const tileOf = (file: GalleryFile) => {
    const media = outputMedia(file, list.data?.workspace ?? workspace.current)
    return { ...media, duration: durations.data?.get(file.name) ?? null }
  }
  const files = $derived(applyEdits(list.data?.files ?? [], edits))
  // A star or a note saved here is held as read (resource's `adopt`), so it
  // is never marked as another writer's change
  function patch(name: string, fields: Partial<GalleryFile>) {
    edits = withPatch(edits, name, fields)
    const held = list.data
    if (held) list.adopt({ ...held, files: applyEdits(held.files, edits) })
  }
  const drop = (names: Iterable<string>) => (edits = withGone(edits, names))
  // The editors' output: completions read the same listing
  $effect(() => {
    const data = list.data
    untrack(() => {
      if (!data) return
      outputLibrary.items = data.files
      outputLibrary.workspace = workspace.current
    })
  })

  const starredOnly = hashState('starred', false, codec.flag)
  // Finals only is where the gallery opens, not a filter: folders still
  // collapse under it
  const filterActive = $derived(
    text.value !== '' ||
      roles.value.join() !== 'final' ||
      starredOnly.value ||
      reqId !== '',
  )
  function clearFilters() {
    text.value = ''
    roles.value = ['final', 'intermediate']
    starredOnly.value = false
    reqId = ''
    reqItem = ''
  }
  const visible = $derived(
    gridFiles(files, {
      text: text.value,
      roles: roles.value,
      starredOnly: starredOnly.value,
      sort: sort.value,
      prompts: list.data?.prompts ?? {},
    }),
  )
  // The headings, and the files in the order they show under them
  const groups = $derived(
    gridGroups(visible, groupBy.value, sort.value, new Date()),
  )
  const shown = $derived(inGroupOrder(visible, groups))
  // A star is undone, not asked about: held here at once (`patch`), as is
  // the Undo that takes it back
  async function setStar(name: string, next: boolean) {
    await api.setStarred(name, next)
    patch(name, { starred: next })
    return { name, next }
  }
  // A star is written in the open workspace, so its Undo is held to it
  const STAR_TARGET = {
    now: () => workspace.current,
    away: 'Nothing undone: another workspace is open now.',
  }
  const starring = action(setStar, {
    doing: 'star that file',
    undo: ({ name, next }) => ({
      message: `${next ? 'Starred' : 'Unstarred'} ${name.split('/').pop()}`,
      revert: () => setStar(name, !next),
      target: STAR_TARGET,
    }),
  })
  const byName = $derived(new Map(visible.map((f) => [f.name, f])))
  // What the viewer steps through: the grid, in its order
  const items = $derived(shown.map(tileOf))

  const downloading = picks.download(api.archiveOutputs, 'files')

  // The question is askDeletion's: it can spare the starred files
  const removingPicked = action(
    async (names: string[]) => {
      const chosen = await askDeletion(files, names)
      if (!chosen) return
      const failures = await actOnEach(chosen, (n) => api.deleteOutput(n))
      const failed = failures.map((f) => f.name)
      const gone = new Set(chosen.filter((name) => !failed.includes(name)))
      drop(gone)
      // Whatever could not be deleted stays selected, so a retry needs no
      // re-ticking and the failure is visible rather than silently dropped
      picks.keepFailed(chosen, failed)
      reportFailures('delete', failures, chosen.length, 'files')
    },
    { doing: 'delete the picked files' },
  )
  const removePicked = () =>
    picks.names.length && removingPicked.run(picks.names)

  // A note changed: carry the file's effective note to its tile, which is
  // what the grid marks. A run's note shows on every tile of the run that
  // inherits it, so that one reads the listing again
  async function noteSaved(
    name: string,
    kind: 'run' | 'file',
    note: string | null,
  ) {
    if (kind === 'run') await list.reload()
    patch(name, { note: note ?? undefined })
  }

  // The viewer steps on from a file that leaves the grid (MediaViewer)
  const removing = action(
    async (file: GalleryFile) => {
      await api.deleteOutput(file.name)
      drop([file.name])
      // A file deleted from here may also be ticked in the grid; leaving it
      // there would make the next bulk action fail on a file that is gone
      picks.drop(file.name)
    },
    { doing: (file) => `delete ${file.name}`, confirm: fileConfirm },
  )
  const saving = action(saveAsAsset, {
    doing: 'save it as an asset',
    success: (r) => `Saved as asset ${r.name}`,
  })

  // An open= that names no file in this view, once the listing is in: said
  // under the header, and cleared. One that was listed and left is the
  // viewer's to step on from
  let missing = $state('')
  let listed: string | null = null
  $effect(() => {
    const name = open
    if (!name || list.status !== 'ready') return
    const here = byName.has(name)
    untrack(() => {
      if (here) listed = name
      else if (name !== listed) {
        missing = leafName(name)
        open = null
      }
    })
  })
  // Another workspace's files are another view: the viewer closes
  let openedIn = workspace.current
  $effect(() => {
    const ws = workspace.current
    untrack(() => {
      if (ws === openedIn) return
      openedIn = ws
      open = null
    })
  })

  // Assess measures shots, seams and a soundtrack; a still has none, and
  // the title says so in the server's own words (dw/server/assess.py)
  const assessKey = (name: string) => AssessCache.key(workspace.current, name)
  const assessable = (kind: string) => kind === 'video' || kind === 'audio'
  const assessTitle = (kind: string) =>
    assessable(kind)
      ? 'measure this file for shots, seams, sync and level - places to look, not verdicts'
      : kind === 'image'
        ? 'a still has no shots, seams or soundtrack to measure'
        : 'not a video or audio file'

  // A bare S: with Shift, Alt or a command modifier it is someone else's
  const isStarKey = (e: KeyboardEvent) =>
    e.key.toLowerCase() === 's' &&
    !(e.metaKey || e.ctrlKey || e.altKey || e.shiftKey)
  const isDeleteKey = (e: KeyboardEvent) =>
    e.key === 'Delete' || (e.key === 'Backspace' && e.metaKey)

  // Culling a run in the viewer: its arrows step, S stars, Delete deletes
  function viewerKey(e: KeyboardEvent, file: MediaFile): boolean {
    const listedFile = byName.get(file.name)
    if (!listedFile) return false
    if (isStarKey(e)) void starring.run(file.name, !listedFile.starred)
    else if (isDeleteKey(e)) void removing.run(listedFile)
    else return false
    return true
  }

  // Escape clears the ticks; the viewer and a dialog answer it themselves
  function onKeydown(e: KeyboardEvent) {
    if (overlayOpen()) return
    if (e.key === 'Escape' && picks.size) picks.clear()
  }
</script>

<svelte:window onkeydown={onKeydown} />

{#snippet search()}
  <input
    class="filter"
    placeholder="search names, prompts and notes…"
    bind:value={text.value}
  />
{/snippet}
<PageHeader
  title="Gallery"
  baseline
  count={{ n: visible.length, of: files.length, noun: 'file' }}
  filter={search}
>
  {#snippet actions()}
    <RoleChips bind:roles={roles.value} bind:starredOnly={starredOnly.value} />
    <RequestFilter bind:request={reqId} bind:item={reqItem} />
    <Segmented
      label="Sort"
      size="sm"
      options={SORT_OPTIONS}
      value={sort.value}
      onselect={(v) => (sort.value = v as GallerySort)}
    />
    <Segmented
      label="Group by"
      size="sm"
      options={GROUP_OPTIONS}
      value={groupBy.value}
      onselect={(v) => (groupBy.value = v as Grouping)}
    />
  {/snippet}
</PageHeader>

{#if missing}
  <p class="hint missing">
    {missing} is not in this view any more.
    <button
      type="button"
      class="quiet icon"
      aria-label="Dismiss"
      title="Dismiss"
      onclick={() => (missing = '')}
    >
      <X size={14} />
    </button>
  </p>
{/if}

<BulkBar
  {picks}
  visible={visible.length}
  {filterActive}
  busy={downloading.busy || removingPicked.busy}
  noun="file"
  ondownload={downloading.run}
  ondelete={removePicked}
  oncompare={() => go('ws', workspace.current, 'compare', ...picks.names)}
/>

<LoadState
  of={list}
  label="the gallery"
  empty={visible.length === 0}
  filtered={files.length > 0}
  onclear={clearFilters}
>
  {#snippet emptyIcon()}<ImageOff size={36} strokeWidth={1.5} />{/snippet}
  {#snippet emptyText()}
    {#if reqId}
      No outputs for this request{reqItem ? ' item' : ''} yet — remove the request
      filter to see everything.
    {:else}
      Nothing generated yet — outputs land here as workflows run.
    {/if}
  {/snippet}
  <!-- Folders and days are shut apart: a day is not a folder -->
  {#key groupBy.value}
    <FolderGroups
      names={shown.map((f) => f.name)}
      {...groups}
      collapseKey="collapsed-gallery-{groupBy.value === 'date'
        ? 'days'
        : 'folders'}"
      {filterActive}
      minColumn="150px"
    >
      {#snippet card(name)}
        {@const file = byName.get(name)!}
        <div
          class="cellwrap"
          class:picked={picks.has(name)}
          class:row-changed={list.changed.has(name)}
        >
          <!-- A sibling of the cell rather than a child: a checkbox nested in
           a button is invalid, and keeping them apart is what lets a plain
           click still open the details it always has -->
          <input
            class="pick"
            type="checkbox"
            checked={picks.has(name)}
            aria-label="select {name}"
            title="select this file for a bulk action"
            onclick={(e) => picks.toggle(name, e.shiftKey)}
          />
          <OutputTile
            file={tileOf(file)}
            caption="clamp"
            active={open === name}
            picked={picks.has(name)}
            onclick={() => (open = name)}
            title="open this file, with what made it"
          >
            {#snippet badges()}
              <RoleTag role={file.role} />
              {#if file.version}
                <span class="version" title={versionTitle(file.version)}>
                  v{file.version}
                </span>
              {/if}
            {/snippet}
          </OutputTile>
        </div>
      {/snippet}
    </FolderGroups>
  {/key}
</LoadState>

{#snippet viewerActions(file: MediaFile)}
  {@const listedFile = byName.get(file.name)}
  {@const assessed = assessCache.get(assessKey(file.name))}
  {#if listedFile}
    <button
      type="button"
      class="quiet icon"
      onclick={() => starring.run(file.name, !listedFile.starred)}
      aria-pressed={!!listedFile.starred}
      aria-label={listedFile.starred ? 'unstar this file' : 'star this file'}
      title={listedFile.starred
        ? 'starred - press S to unstar'
        : 'star this file - press S'}
    >
      <Star size={16} fill={listedFile.starred ? 'currentColor' : 'none'} />
    </button>
    <button
      type="button"
      class="quiet withicon"
      onclick={() => saving.run(listedFile)}
      disabled={saving.busy}
      title="save this file to the asset library, under a name later workflows can use"
    >
      <Bookmark size={14} />Save as asset
    </button>
    {#if file.kind === 'image' || file.kind === 'video'}
      <button
        type="button"
        class="quiet withicon"
        onclick={() => {
          useInput = file
          useInputOpen = true
        }}
        title="open a workflow of this workspace with this file filled in"
      >
        <ImagePlus size={14} />Use as input…
      </button>
    {/if}
    <ReferenceChip prefix={OUTPUT} name={file.name} />
    <button
      type="button"
      class="quiet withicon"
      onclick={() => assessCache.run(assessKey(file.name), file.name)}
      disabled={!assessable(file.kind) || assessed?.running}
      title={assessTitle(file.kind)}
    >
      <Gauge size={14} />{assessed?.running ? 'Assessing…' : 'Assess'}
    </button>
    <DownloadLink href={api.outputDownloadUrl(file.name)} />
    <button
      type="button"
      class="quiet icon danger"
      onclick={() => removing.run(listedFile)}
      title="delete this file from the output directory"
      aria-label="delete this file from the output directory"
    >
      <Trash2 size={16} />
    </button>
  {/if}
{/snippet}

{#snippet viewerAside(file: MediaFile)}
  {@const listedFile = byName.get(file.name)}
  <!-- Keyed by the file, so each starts its own read and its own note box -->
  {#key file.name}
    <RecipePanel name={file.name} source="output" workspace={file.workspace}>
      {#snippet children(meta, reload)}
        {#if listedFile}
          <GalleryAside
            file={listedFile}
            {meta}
            {reload}
            workspace={file.workspace}
            assessed={assessCache.get(assessKey(file.name))}
            onnotesaved={(kind, note) => noteSaved(file.name, kind, note)}
          />
        {/if}
      {/snippet}
    </RecipePanel>
  {/key}
{/snippet}

<UseAsInput file={useInput} bind:open={useInputOpen} />

<MediaViewer
  {items}
  bind:open
  actions={viewerActions}
  aside={viewerAside}
  onkey={viewerKey}
/>

<style>
  .missing {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin: 0 0 var(--space-2);
  }
  /* The one mark that tells four runs' same-named files apart */
  .version {
    display: inline-block;
    padding: 0 0.3rem;
    border-radius: 0.2rem;
    background: var(--line);
    color: var(--ink);
    font-weight: 600;
  }
</style>
