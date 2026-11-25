<script lang="ts">
  import { Gauge, ImageOff, Trash2 } from '@lucide/svelte'
  import { outputLibrary } from '../outputlib.svelte'
  import { api } from '../api'
  import DownloadLink from '../DownloadLink.svelte'
  import {
    outputMedia,
    viewLabels,
    withoutRun,
    type MediaFile,
    durationsByName,
  } from '../media'
  import { isStarKey, openInHash } from '../mediaViewer'
  import MediaViewer from '../ui/MediaViewer.svelte'
  import MissingFile from '../ui/MissingFile.svelte'
  import { missingFile } from '../missingFile.svelte'
  import RecipePanel from '../ui/RecipePanel.svelte'
  import { untrack } from 'svelte'
  import { resource, type Resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import FolderGroups from '../FolderGroups.svelte'
  import BulkBar from '../BulkBar.svelte'
  import {
    Picks,
    STAR_CHANGED,
    actOnEach,
    reportFailures,
  } from '../picks.svelte'
  import { notify } from '../toast'
  import { overlayOpen } from '../ui/layers.svelte'
  import { action } from '../action.svelte'
  import { askDeletion, fileConfirm } from '../gallery/deletion'
  import type { GalleryFile, OutputRole } from '../types'
  import { workspace } from '../workspace.svelte'
  import { codec, goHref, hashState } from '../router.svelte'
  import { compareHref, wsHref } from '../routes'
  import { collapsedState } from '../collapsed.svelte'
  import GalleryFind, { findState } from '../gallery/GalleryFind.svelte'
  import GalleryCell from '../gallery/GalleryCell.svelte'
  import RequestFilter from '../gallery/RequestFilter.svelte'
  import RoleChips from '../gallery/RoleChips.svelte'
  import GalleryAside from '../gallery/GalleryAside.svelte'
  import { SAVING_ASSET, saveAsAsset } from '../gallery/saveAsAsset'
  import UseAsInput from '../gallery/UseAsInput.svelte'
  import ReuseActions from '../gallery/ReuseActions.svelte'
  import {
    AssessCache,
    assessable,
    assessTitle,
  } from '../gallery/assess.svelte'
  import {
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
    type Listing,
  } from '../gallery/order'

  // The filters live in the hash, so opening a file and pressing Back - or
  // sharing the link - keeps the narrowed view
  const find = findState()
  const { text, sort, groupBy } = find
  // The roles to show. Finals are what a run was for, so the gallery opens
  // on them; Intermediate adds the steps' working files
  const roles = hashState<OutputRole[]>(
    'roles',
    ['final'],
    codec.someOf<OutputRole>(['final', 'intermediate'], ['final']),
  )
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
      return durationsByName(result.files)
    },
    { watch: ['gallery'] },
  )
  const files = $derived(applyEdits(list.data?.files ?? [], edits))
  // Same-leaf files in different folders are told apart by their caption
  const labelled = $derived(
    viewLabels(
      files.map((f) => f.name),
      withoutRun(files),
    ),
  )
  const tileOf = (file: GalleryFile) => {
    const media = outputMedia(file, list.data?.workspace ?? workspace.current)
    const duration = durations.data?.get(file.name) ?? null
    return labelled({ ...media, duration })
  }
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
  // Finals, with or without the intermediates, hide nothing a run was
  // for: not a filter, so folders still collapse under them
  const filterActive = $derived(
    text.value !== '' ||
      !roles.value.includes('final') ||
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
  // The headings, and the files in the order they show under them. Folders
  // and days are shut apart (a day is not a folder), and held here so a
  // shut one's files are out of the picks' and the viewer's order too
  const groups = $derived(
    gridGroups(visible, groupBy.value, sort.value, new Date()),
  )
  const shutBy = {
    folder: collapsedState('collapsed-gallery-folders'),
    date: collapsedState('collapsed-gallery-days'),
  }
  const collapsed = $derived(shutBy[groupBy.value])
  const shown = $derived(
    inGroupOrder(visible, groups, (g) => !filterActive && collapsed.has(g)),
  )
  // A star is undone, not asked about: held here at once (`patch`), as is
  // the Undo that takes it back - unless the listing says another writer
  // changed the star since, which the Undo leaves
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
      revert: async () => {
        if (!!byName.get(name)?.starred !== next)
          return notify.error(STAR_CHANGED)
        return setStar(name, !next)
      },
      target: STAR_TARGET,
    }),
  })
  // Every file's star as the grid holds it, a filter's hidden ones
  // included: Undo restores each as it was, and leaves one changed since
  const byName = $derived(new Map(files.map((f) => [f.name, f])))
  const starringPicked = picks.star(api.setStarred, 'file', {
    was: (name) => !!byName.get(name)?.starred,
    applied: (name, next) => patch(name, { starred: next }),
    target: STAR_TARGET,
  })
  const allStarred = $derived(
    picks.names.length > 0 && picks.names.every((n) => byName.get(n)?.starred),
  )
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
  const saving = action(saveAsAsset, SAVING_ASSET)

  // An open= that names no file in this view, once the listing is in: said
  // under the header (missingFile.svelte.ts), and cleared
  const stray = missingFile({
    open: () => open,
    known: () => list.status === 'ready',
    names: () => files.map((f) => f.name),
  })
  $effect(() => void (stray.missing && untrack(() => (open = null))))
  // A link to a file in a shut folder or day opens the group too, so the
  // viewer's count and the grid agree; once, so the person can shut it
  let revealed: string | null = null
  $effect(() => {
    const name = open
    if (!name || !byName.has(name)) return
    untrack(() => {
      const g = groups.groupOf(name)
      if (name !== revealed && !filterActive && collapsed.has(g))
        collapsed.toggle(g)
      revealed = name
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

  const assessKey = (name: string) => AssessCache.key(workspace.current, name)

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

<GalleryFind {find} count={{ n: visible.length, of: files.length }}>
  <RoleChips bind:roles={roles.value} bind:starredOnly={starredOnly.value} />
  <RequestFilter bind:request={reqId} bind:item={reqItem} />
</GalleryFind>

<MissingFile leaf={stray.missing} ondismiss={stray.dismiss} />

<BulkBar
  {picks}
  visible={shown.length}
  {filterActive}
  busy={downloading.busy || removingPicked.busy || starringPicked.busy}
  {allStarred}
  onstar={starringPicked.run}
  noun="file"
  ondownload={downloading.run}
  ondelete={removePicked}
  oncompare={() => goHref(compareHref(workspace.current, picks.names))}
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
      Nothing generated yet.
      <a href={wsHref(workspace.current, 'workflows')}>Run a workflow</a> and what
      it makes lands here.
    {/if}
  {/snippet}
  <!-- Remounted per grouping: FolderGroups reads its shut state once -->
  {#key groupBy.value}
    <FolderGroups
      names={visible.map((f) => f.name)}
      {...groups}
      {collapsed}
      {filterActive}
      minColumn="150px"
    >
      {#snippet card(name)}
        <GalleryCell
          file={byName.get(name)!}
          tile={tileOf(byName.get(name)!)}
          {picks}
          changed={list.changed.has(name)}
          active={open === name}
          onopen={() => (open = name)}
        />
      {/snippet}
    </FolderGroups>
  {/key}
</LoadState>

{#snippet viewerActions(file: MediaFile)}
  {@const listedFile = byName.get(file.name)}
  {@const assessed = assessCache.get(assessKey(file.name))}
  {#if listedFile}
    <ReuseActions
      {file}
      starred={!!listedFile.starred}
      saving={saving.busy}
      onstar={() => starring.run(file.name, !listedFile.starred)}
      onsave={() => saving.run(listedFile)}
      onuse={() => {
        useInput = file
        useInputOpen = true
      }}
    />
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
      title="Delete this file from the output directory"
      aria-label="Delete"
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
  ongone={stray.gone}
/>
