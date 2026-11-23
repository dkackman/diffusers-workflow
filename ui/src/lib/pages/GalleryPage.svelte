<script lang="ts">
  import { ImageOff, Pencil, Play, Star } from '@lucide/svelte'
  import { outputLibrary } from '../outputlib.svelte'
  import { api } from '../api'
  import { untrack } from 'svelte'
  import { resource, type Resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import FolderGroups from '../FolderGroups.svelte'
  import BulkBar from '../BulkBar.svelte'
  import { Picks, actOnEach, reportFailures } from '../picks.svelte'
  import { isEditable } from '../editable'
  import { neighbor } from '../neighbor'
  import { overlayOpen } from '../ui/layers.svelte'
  import { notify } from '../toast'
  import { action } from '../action.svelte'
  import { askDeletion, fileConfirm } from '../gallery/deletion'
  import type { GalleryFile, OutputRole } from '../types'
  import { workspace } from '../workspace.svelte'
  import { go } from '../router.svelte'
  import RequestFilter from '../gallery/RequestFilter.svelte'
  import RoleChips from '../gallery/RoleChips.svelte'
  import GalleryDetail from '../gallery/GalleryDetail.svelte'
  import { AssessCache } from '../gallery/assess.svelte'
  import RoleTag from '../gallery/RoleTag.svelte'
  import {
    applyEdits,
    gridFiles,
    keepPicks,
    listingRows,
    noEdits,
    settleEdits,
    withGone,
    withPatch,
    type Listing,
  } from '../gallery/order'

  let filter = $state('')
  // The roles to show. Finals are what a run was for, so the gallery opens
  // on them; Intermediate adds the steps' working files
  let roles = $state<OutputRole[]>(['final'])
  let selected = $state<GalleryFile | null>(null)
  // Names ticked in the grid, for the bulk actions. The grid order it is
  // given is what a shift-range spans and the order the actions act in
  const picks: Picks = new Picks(() => visible.map((f) => f.name))
  let busy = $state(false)
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
      )
      return { files: result.files, key, began }
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

  let starredOnly = $state(false)
  const filterActive = $derived(
    filter !== '' || roles.length < 2 || starredOnly || reqId !== '',
  )
  function clearFilters() {
    filter = ''
    roles = ['final', 'intermediate']
    starredOnly = false
    reqId = ''
    reqItem = ''
  }
  const visible = $derived(gridFiles(files, filter, roles, starredOnly))
  async function toggleStar() {
    if (!selected) return
    const name = selected.name
    const next = !selected.starred
    try {
      await api.setStarred(name, next)
      patch(name, { starred: next })
      if (selected?.name === name) selected = { ...selected, starred: next }
    } catch (e) {
      notify.failure('star that file', e)
    }
  }
  const byName = $derived(new Map(visible.map((f) => [f.name, f])))
  // The server folds each run id into `folder`; group by that, since the
  // name alone reads the run directory as one more folder
  const folderByName = $derived(new Map(visible.map((f) => [f.name, f.folder])))

  async function downloadPicked() {
    if (picks.names.length === 0) return
    busy = true
    try {
      await api.archiveOutputs(picks.names)
    } catch (e) {
      notify.failure('download the picked files', e)
    } finally {
      busy = false
    }
  }

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
      if (selected && gone.has(selected.name)) selected = null
      reportFailures('delete', failures, chosen.length, 'files')
    },
    { doing: 'delete the picked files' },
  )
  const removePicked = () =>
    picks.names.length && removingPicked.run(picks.names)

  // Where the open file stood in the grid when last seen there, so stepping
  // goes on from that place once a filter or an unstar has hidden it
  let selectedAt = -1
  // The grid's order, noting the open file's place while it is still listed
  function gridOrder() {
    const order = visible.map((f) => f.name)
    const at = selected ? order.indexOf(selected.name) : -1
    if (at !== -1) selectedAt = at
    return order
  }

  function select(file: GalleryFile) {
    selected = file
    selectedAt = visible.findIndex((f) => f.name === file.name)
  }

  // A note changed: carry the file's effective note to its tile, which is
  // what the grid marks. A run's note shows on every tile of the run that
  // inherits it, so that one reads the listing again
  async function noteSaved(kind: 'run' | 'file', note: string | null) {
    if (!selected) return
    const name = selected.name
    if (kind === 'run') await list.reload()
    patch(name, { note: note ?? undefined })
  }

  const removing = action(
    async (file: GalleryFile) => {
      const name = file.name
      const order = gridOrder()
      const after =
        neighbor(order, name, 1, selectedAt) ??
        neighbor(order, name, -1, selectedAt)
      await api.deleteOutput(name)
      drop([name])
      // A file deleted from here may also be ticked in the grid; leaving it
      // there would make the next bulk action fail on a file that is gone
      picks.drop(name)
      // Culling goes on to the next file rather than back to the grid
      if (after && byName.get(after)) select(byName.get(after)!)
      else selected = null
    },
    { doing: (file) => `delete ${file.name}`, confirm: fileConfirm },
  )
  const removeFile = () => selected && removing.run(selected)

  // Escape closes the thing on top, and a dialog answers it itself. The
  // selection is the more recent, more surprising state to be stuck in,
  // so it clears first and the detail panel on a second press
  function escape() {
    if (picks.size) picks.clear()
    else selected = null
  }

  function step(e: KeyboardEvent, by: 1 | -1) {
    if (!selected) return
    const next = neighbor(gridOrder(), selected.name, by, selectedAt)
    const file = next ? byName.get(next) : undefined
    if (!file) return
    e.preventDefault()
    select(file)
  }

  // A bare S: with Shift, Alt or a command modifier it is someone else's
  const isStarKey = (e: KeyboardEvent) =>
    e.key.toLowerCase() === 's' &&
    !(e.metaKey || e.ctrlKey || e.altKey || e.shiftKey)

  // Culling a run: step through what the grid shows, delete as you go
  function cullKey(e: KeyboardEvent) {
    if (e.key === 'ArrowRight') step(e, 1)
    else if (e.key === 'ArrowLeft') step(e, -1)
    else if (e.key === 'Delete' || (e.key === 'Backspace' && e.metaKey)) {
      e.preventDefault()
      void removeFile()
    } else if (isStarKey(e)) {
      e.preventDefault()
      void toggleStar()
    }
  }

  function onKeydown(e: KeyboardEvent) {
    if (overlayOpen()) return
    if (e.key === 'Escape') return escape()
    // A key typed into the note box, or a focused video seeking, is theirs
    if (selected && !isEditable(e.target)) cullKey(e)
  }
</script>

<svelte:window onkeydown={onKeydown} />

{#snippet search()}
  <input class="filter" placeholder="filter…" bind:value={filter} />
{/snippet}
<PageHeader
  title="Gallery"
  baseline
  count={{ n: files.length, noun: 'file' }}
  filter={search}
>
  {#snippet actions()}
    <RoleChips bind:roles bind:starredOnly />
    <RequestFilter bind:request={reqId} bind:item={reqItem} />
  {/snippet}
</PageHeader>

<BulkBar
  {picks}
  visible={visible.length}
  {filterActive}
  busy={busy || removingPicked.busy}
  noun="file"
  ondownload={downloadPicked}
  ondelete={removePicked}
  oncompare={() => go('ws', workspace.current, 'compare', ...picks.names)}
/>

<LoadState
  of={list}
  label="the gallery"
  empty={visible.length === 0}
  filtered={filterActive && files.length > 0}
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
  <FolderGroups
    names={visible.map((f) => f.name)}
    groupOf={(name) => folderByName.get(name) ?? ''}
    collapseKey="collapsed-gallery-folders"
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
        <button
          class="cell"
          class:active={selected?.name === name}
          onclick={() => select(file)}
          title="show details{file.kind === 'image'
            ? ' and generation metadata'
            : ''}"
        >
          {#if file.kind === 'image'}
            <img
              src={api.galleryThumbnailUrl(file.name)}
              alt={file.name}
              loading="lazy"
            />
          {:else if file.kind === 'video'}
            <!-- a poster, never a <video>: a grid of clips opening a stream
               per tile hung the page -->
            <span class="poster">
              <img
                src={api.galleryThumbnailUrl(file.name)}
                alt={file.name}
                loading="lazy"
              />
              <span class="videobadge" aria-hidden="true">
                <Play size={11} />
              </span>
            </span>
          {:else}
            <span class="audio">
              {file.kind === 'text' ? '¶' : '♪'}
              {file.label}
            </span>
          {/if}
          <span class="caption" title={file.name}>
            {#if file.role}
              <RoleTag role={file.role} />
              <!-- eslint-disable-next-line svelte/no-useless-mustaches -->
              {' '}
            {/if}
            {#if file.version}
              <span
                class="version"
                title="version {file.version} of this workflow"
              >
                v{file.version}
              </span>
              <!-- A real space rather than a margin, so a screen reader says
                 "v4 film.mp4" instead of running the two together. Explicit:
                 whitespace at the edge of an if block is trimmed -->
              <!-- eslint-disable-next-line svelte/no-useless-mustaches -->
              {' '}
            {/if}{file.label}{#if file.note}
              <!-- eslint-disable-next-line svelte/no-useless-mustaches -->
              {' '}<span
                class="note-mark"
                title={file.note.split('\n')[0]}
                aria-label="Has a note"
              >
                <Pencil size={11} />
              </span>{/if}{#if file.starred}
              <!-- eslint-disable-next-line svelte/no-useless-mustaches -->
              {' '}<span class="note-mark" aria-label="Starred">
                <Star size={11} fill="currentColor" />
              </span>{/if}
          </span>
        </button>
      </div>
    {/snippet}
  </FolderGroups>
</LoadState>

{#if selected}
  <!-- Keyed by the pick, so each file's panel starts its own read -->
  {#key selected.name}
    <GalleryDetail
      file={selected}
      {assessCache}
      onnotesaved={noteSaved}
      starred={!!selected.starred}
      onstar={toggleStar}
      onremove={removeFile}
      onclose={() => (selected = null)}
    />
  {/key}
{/if}

<style>
  .note-mark {
    color: var(--muted);
    vertical-align: -1px;
  }
  .cellwrap {
    display: flex;
  }
  /* A frame with a caption strip under it, like the catalog's cards: the
     picture bleeds to the edges and the label sits below the rule rather
     than floating on a padded card */
  .cell {
    flex: 1;
    min-width: 0;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-2);
    padding: 0;
    overflow: hidden;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    color: var(--muted);
    font-weight: 500;
    font-size: var(--t-xs);
    text-align: left;
    /* A folder of thousands of outputs still renders every cell's DOM node
       (no true windowing library is in the project's dependencies yet),
       but content-visibility skips layout/paint for cells scrolled out of
       view, which is most of the win for cheap. contain-intrinsic-size
       keeps scrollbar height stable before a cell has ever been measured. */
    content-visibility: auto;
    contain-intrinsic-size: 150px 210px;
  }
  .cell:hover,
  .cell.active {
    border-color: var(--ink);
    filter: none;
  }
  .poster {
    display: block;
    position: relative;
  }
  .cell img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    display: block;
  }
  .audio {
    aspect-ratio: 1;
    display: grid;
    place-items: center;
    font-family: var(--font-mono);
    background: var(--panel-2);
  }
  /* The file's own name, which is what the engine wrote and what you would
     type to reference it */
  .caption {
    font-family: var(--font-mono);
    padding: 0.35rem 0.5rem 0.4rem;
    border-top: 1px solid var(--line);
    overflow: hidden;
    /* Sibling outputs of one step share a long common prefix and differ only
       near the end (the i.j.k index, or a dedupe counter right before the
       extension) - a single nowrap+ellipsis line would hide exactly the part
       that tells them apart, so wrap onto a few lines and break mid-token
       instead of clipping. */
    display: -webkit-box;
    -webkit-line-clamp: 4;
    line-clamp: 4;
    -webkit-box-orient: vertical;
    white-space: normal;
    word-break: break-all;
  }
  /* The one part of the caption that must not be broken or clamped away:
     with four runs writing the same name it is the only thing on the card
     that differs. Inline-block so word-break: break-all cannot split 'v10'
     across lines */
  .version {
    display: inline-block;
    padding: 0 0.3rem;
    border-radius: 0.2rem;
    background: var(--line);
    color: var(--ink);
    font-weight: 600;
    word-break: keep-all;
  }
</style>
