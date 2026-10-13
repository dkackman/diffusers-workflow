<script lang="ts">
  import { ImageOff, Play } from '@lucide/svelte'
  import { outputLibrary } from '../outputlib.svelte'
  import { api } from '../api'
  import Empty from '../Empty.svelte'
  import FolderGroups from '../FolderGroups.svelte'
  import BulkBar from '../BulkBar.svelte'
  import { Picks, actOnEach } from '../picks.svelte'
  import { overlayOpen } from '../ui/layers.svelte'
  import { notify } from '../toast'
  import { confirmDialog } from '../confirm.svelte'
  import type { GalleryFile, GalleryInput, OutputRole } from '../types'
  import { workspace } from '../workspace.svelte'
  import GalleryDetail from '../gallery/GalleryDetail.svelte'
  import RoleTag from '../gallery/RoleTag.svelte'

  let files = $state<GalleryFile[]>([])
  let loaded = $state(false)
  let filter = $state('')
  // The roles to show. Finals are what a run was for, so the gallery opens
  // on them; Intermediate adds the steps' working files
  let roles = $state<OutputRole[]>(['final'])
  let error = $state('')
  let selected = $state<GalleryFile | null>(null)
  // Names ticked in the grid, for the bulk actions. The grid order it is
  // given is what a shift-range spans and the order the actions act in
  const picks = new Picks(() => visible.map((f) => f.name))
  let busy = $state(false)
  let metadata = $state<Record<string, unknown> | null>(null)
  let metadataLoading = $state(false)
  let sourceJob = $state<{ id: string; status: string } | null>(null)
  let role = $state<OutputRole | null>(null)
  let inputs = $state<GalleryInput[]>([])

  $effect(() => {
    // Read inside the effect so switching workspaces refetches the gallery
    void workspace.current
    api
      .gallery()
      .then((result) => {
        files = result.files
        // The editors' output: completions read the same listing
        outputLibrary.items = result.files
        outputLibrary.workspace = workspace.current
        error = ''
        picks.keepOnly(result.files.map((f) => f.name))
      })
      .catch((e) => (error = e.message))
      .finally(() => (loaded = true))
  })

  // A file whose run does not say its role counts as final
  const roleOf = (f: GalleryFile): OutputRole => f.role ?? 'final'
  const filterActive = $derived(filter !== '' || roles.length < 2)
  // Finals lead inside a run only: the server's order (newest first) holds
  // between runs, so runs sit where their newest file put them. The sort is
  // stable, so it holds inside each role too
  const runKey = (f: GalleryFile) => `${f.folder}\u0000${f.run_id}`
  const visible = $derived.by(() => {
    const shown = files.filter(
      (f) =>
        f.name.toLowerCase().includes(filter.toLowerCase()) &&
        roles.includes(roleOf(f)),
    )
    const runs: string[] = []
    for (const f of shown) if (!runs.includes(runKey(f))) runs.push(runKey(f))
    const rank = (f: GalleryFile) => Number(roleOf(f) === 'intermediate')
    return shown.sort(
      (a, b) =>
        runs.indexOf(runKey(a)) - runs.indexOf(runKey(b)) || rank(a) - rank(b),
    )
  })
  // Never both off: an empty grid is not a state worth offering
  function toggleRole(which: OutputRole) {
    const next = roles.includes(which)
      ? roles.filter((r) => r !== which)
      : [...roles, which]
    roles = next.length ? next : [which === 'final' ? 'intermediate' : 'final']
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
        `Delete ${names.length} file${names.length === 1 ? '' : 's'}? This removes them on disk.`,
        { confirmLabel: 'Delete' },
      ))
    )
      return
    busy = true
    const failed = await actOnEach(names, (name) => api.deleteOutput(name))
    const gone = new Set(names.filter((name) => !failed.includes(name)))
    files = files.filter((f) => !gone.has(f.name))
    // Whatever could not be deleted stays selected, so a retry needs no
    // re-ticking and the failure is visible rather than silently dropped
    picks.keepFailed(names, failed)
    if (selected && gone.has(selected.name)) selected = null
    if (failed.length)
      notify.error(
        `Could not delete ${failed.length} of ${names.length} files: ${failed.join(', ')}`,
      )
    busy = false
  }

  function select(file: GalleryFile) {
    selected = file
    metadata = null
    metadataLoading = true
    sourceJob = null
    role = null
    inputs = []
    api
      .galleryMetadata(file.name)
      .then((r) => {
        if (selected?.name === file.name) {
          metadata = r.metadata
          sourceJob = r.job
          role = r.role ?? file.role ?? null
          inputs = r.inputs ?? []
        }
      })
      .catch(() => {})
      .finally(() => {
        if (selected?.name === file.name) metadataLoading = false
      })
  }

  async function removeFile() {
    if (!selected) return
    if (
      !(await confirmDialog(
        `Delete ${selected.name}? This removes the file on disk.`,
        { confirmLabel: 'Delete' },
      ))
    )
      return
    const name = selected.name
    try {
      await api.deleteOutput(name)
      files = files.filter((f) => f.name !== name)
      // A file deleted from here may also be ticked in the grid; leaving it
      // there would make the next bulk action fail on a file that is gone
      picks.drop(name)
      selected = null
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      notify.error(msg)
    }
  }
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

<div class="pagehead baseline">
  <h1>Gallery</h1>
  <span class="num muted">{files.length} files</span>
  <input class="filter" placeholder="filter…" bind:value={filter} />
  <div class="rolechips" role="group" aria-label="filter by role">
    {#each ['final', 'intermediate'] as const as which (which)}
      <button
        class="rolechip"
        class:on={roles.includes(which)}
        aria-pressed={roles.includes(which)}
        onclick={() => toggleRole(which)}
        title={which === 'final'
          ? 'what the runs were for'
          : "the steps' working files, kept along the way"}
        >{which === 'final' ? 'Final' : 'Intermediate'}</button
      >
    {/each}
  </div>
</div>

{#if error}<p class="muted">Could not read the gallery: {error}</p>{/if}
{#if loaded && !error && files.length === 0}
  <Empty>
    {#snippet icon()}<ImageOff size={36} strokeWidth={1.5} />{/snippet}
    Nothing generated yet — outputs land here as workflows run.
  </Empty>
{/if}

<BulkBar
  {picks}
  visible={visible.length}
  {filterActive}
  {busy}
  noun="file"
  ondownload={downloadPicked}
  ondelete={removePicked}
/>

<FolderGroups
  names={visible.map((f) => f.name)}
  groupOf={(name) => folderByName.get(name) ?? ''}
  collapseKey="collapsed-gallery-folders"
  {filterActive}
  minColumn="150px"
>
  {#snippet card(name)}
    {@const file = byName.get(name)!}
    <div class="cellwrap" class:picked={picks.has(name)}>
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
            <span class="videobadge" aria-hidden="true"><Play size={11} /></span
            >
          </span>
        {:else}
          <span class="audio"
            >{file.kind === 'text' ? '¶' : '♪'} {file.label}</span
          >
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
              >v{file.version}</span
            >
            <!-- A real space rather than a margin, so a screen reader says
                 "v4 film.mp4" instead of running the two together. Explicit:
                 whitespace at the edge of an if block is trimmed -->
            <!-- eslint-disable-next-line svelte/no-useless-mustaches -->
            {' '}
          {/if}{file.label}</span
        >
      </button>
    </div>
  {/snippet}
</FolderGroups>

{#if selected}
  <GalleryDetail
    file={selected}
    {metadata}
    {metadataLoading}
    {sourceJob}
    {role}
    {inputs}
    onremove={removeFile}
    onclose={() => (selected = null)}
  />
{/if}

<style>
  .filter {
    max-width: 220px;
    margin-left: auto;
  }
  .rolechips {
    display: flex;
    gap: 0.3rem;
  }
  .rolechip {
    background: none;
    border: 1px solid var(--line);
    border-radius: 999px;
    color: var(--muted);
    font-size: var(--t-xs);
    font-weight: 500;
    padding: 0.05rem 0.6rem;
    cursor: pointer;
  }
  .rolechip:hover {
    color: var(--ink);
    border-color: var(--muted);
    filter: none;
  }
  .rolechip.on {
    background: var(--ink);
    border-color: var(--ink);
    color: var(--panel);
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
