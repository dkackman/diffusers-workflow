<script lang="ts">
  import { Bookmark, ImageOff, FolderOpen, Trash2, X } from '@lucide/svelte'
  import DownloadLink from '../DownloadLink.svelte'
  import { api } from '../api'
  import Empty from '../Empty.svelte'
  import FolderGroups from '../FolderGroups.svelte'
  import { goWs } from '../router.svelte'
  import BulkBar from '../BulkBar.svelte'
  import { Picks, actOnEach, dialogOpen } from '../picks.svelte'
  import { notify } from '../toast'
  import { confirmDialog } from '../confirm.svelte'
  import type { GalleryFile } from '../types'
  import { workspace } from '../workspace.svelte'
  import { formatBytes, formatMtime } from '../format'

  let files = $state<GalleryFile[]>([])
  let loaded = $state(false)
  let filter = $state('')
  // The in-run subfolder to show - null is every one. Only offered when
  // some output landed in one; a workspace of unfoldered runs never sees
  // the control
  let subfolder = $state<string | null>(null)
  let error = $state('')
  let selected = $state<GalleryFile | null>(null)
  // Names ticked in the grid, for the bulk actions. The grid order it is
  // given is what a shift-range spans and the order the actions act in
  const picks = new Picks(() => visible.map((f) => f.name))
  let busy = $state(false)
  let metadata = $state<Record<string, unknown> | null>(null)
  let metadataLoading = $state(false)
  let sourceJob = $state<{ id: string; status: string } | null>(null)

  $effect(() => {
    // Read inside the effect so switching workspaces refetches the gallery
    void workspace.current
    api
      .gallery()
      .then((result) => {
        files = result.files
        error = ''
        picks.keepOnly(result.files.map((f) => f.name))
      })
      .catch((e) => (error = e.message))
      .finally(() => (loaded = true))
  })

  /** Keep this file as an input asset, so a later workflow can name it
   * without depending on the run that made it. */
  async function keepAsAsset() {
    if (!selected) return
    const suggestion = selected.name.split('/').pop() ?? selected.name
    const assetName = window.prompt(
      'Keep as asset — name in the asset library:',
      suggestion,
    )
    if (!assetName) return
    try {
      const result = await api.keepOutput(selected.name, assetName)
      notify.success(`Kept as ${result.reference}`)
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e)
      // The server refuses an existing name rather than replacing it; the
      // choice to replace belongs to the person, not the button
      if (
        message.includes('already exists') &&
        (await confirmDialog(`${message}\n\nReplace it?`, {
          confirmLabel: 'Replace',
        }))
      ) {
        try {
          const result = await api.keepOutput(selected.name, assetName, true)
          notify.success(`Kept as ${result.reference}`)
        } catch (failure) {
          notify.error(
            failure instanceof Error ? failure.message : String(failure),
          )
        }
      } else {
        notify.error(message)
      }
    }
  }

  // Every distinct subfolder in the listing - read off the entries so a
  // delete updates it without a refetch. '' (the run root) appears only
  // when some file sits there, so the pick never offers an empty option
  const subfolders = $derived(
    [...new Set(files.map((f) => f.subfolder))].sort((a, b) =>
      a.localeCompare(b),
    ),
  )
  const subfolderOffered = $derived(subfolders.some((s) => s !== ''))
  const filterActive = $derived(filter !== '' || subfolder !== null)
  const visible = $derived(
    files.filter(
      (f) =>
        f.name.toLowerCase().includes(filter.toLowerCase()) &&
        (subfolder === null || f.subfolder === subfolder),
    ),
  )
  // A pick that no longer exists (its last file deleted) means everything,
  // not an empty grid pinned to a vanished value - and the control itself
  // is reset, since a <select> whose value matches no option shows blank.
  // A pick of '' (the run root) never appears in `subfolders` on its own,
  // so it is only cleared once the control itself stops being offered -
  // otherwise it would survive the control unmounting and leave
  // `filterActive` stuck true with nothing to reset it
  $effect(() => {
    if (
      subfolder !== null &&
      (!subfolderOffered || !subfolders.includes(subfolder))
    )
      subfolder = null
  })
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
    api
      .galleryMetadata(file.name)
      .then((r) => {
        if (selected?.name === file.name) {
          metadata = r.metadata
          sourceJob = r.job
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

  const embeddedWorkflow = $derived(
    (metadata?.workflow as Record<string, unknown> | undefined) ?? null,
  )

  // The step's realized arguments, so the prompt shown is the text the
  // pipeline actually saw rather than the 'variable:' reference in the JSON
  const args = $derived(
    (metadata?.arguments as Record<string, unknown> | undefined) ?? null,
  )

  /** A prompt argument as displayable text - pipelines accept a list too. */
  function promptText(value: unknown): string {
    if (typeof value === 'string') return value
    if (Array.isArray(value))
      return value.filter((v) => typeof v === 'string').join('\n')
    return ''
  }

  const prompt = $derived(promptText(args?.prompt))
  const negativePrompt = $derived(promptText(args?.negative_prompt))
  const seed = $derived(
    typeof metadata?.seed === 'number' ? metadata.seed : args?.seed,
  )

  function openAsWorkflow() {
    if (!embeddedWorkflow) return
    // Pin the run's seed into the definition so reopening reproduces this
    // exact image; delete the seed field in the editor to re-randomize
    const definition = { ...embeddedWorkflow }
    if (typeof metadata?.seed === 'number') definition.seed = metadata.seed
    sessionStorage.setItem('dw-editor-import', JSON.stringify(definition))
    goWs('edit')
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
  <h1>Gallery</h1>
  <span class="num muted">{files.length} files</span>
  <input class="filter" placeholder="filter…" bind:value={filter} />
  {#if subfolderOffered}
    <select
      class="subfolderpick"
      aria-label="subfolder"
      title="show only files a step saved to this subfolder of its run"
      bind:value={subfolder}
    >
      <option value={null}>all subfolders</option>
      {#each subfolders as s (s)}
        <option value={s}>{s === '' ? '(run root)' : `${s}/`}</option>
      {/each}
    </select>
  {/if}
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
          <video src={file.url} preload="metadata" muted></video>
        {:else}
          <span class="audio">♪ {file.label}</span>
        {/if}
        <span class="caption" title={file.name}>
          {#if file.version}
            <span
              class="version"
              title="version {file.version} of this workflow"
              >v{file.version}</span
            >
          {/if}{file.label}</span
        >
      </button>
    </div>
  {/snippet}
</FolderGroups>

{#if selected}
  <div class="detail panel">
    <div class="bar">
      <strong class="selname">{selected.name}</strong>
      <span class="flex"></span>
      {#if embeddedWorkflow}
        <button
          class="withicon"
          onclick={openAsWorkflow}
          title="open the embedded workflow definition in the editor"
        >
          <FolderOpen size={14} />Open as workflow
        </button>
      {/if}
      <button
        class="withicon"
        onclick={keepAsAsset}
        title="keep this file as an input asset, under a name later workflows can use"
      >
        <Bookmark size={14} />Keep as asset
      </button>
      <a
        href={selected.url}
        target="_blank"
        class="muted"
        title="open the file itself in a new tab">open file</a
      >
      {#if selected.version}
        <!-- The run this file came from, said in both the form a person is
             quoted ("version 4") and the form every tool takes (the run
             id), so the two can be checked against each other here rather
             than back in the listing -->
        <span class="num muted"
          >version {selected.version} · <code>{selected.run_id}</code></span
        >
      {/if}
      <span class="num muted"
        >{formatBytes(selected.size)} · {formatMtime(selected.mtime)}</span
      >
      <DownloadLink href={api.outputDownloadUrl(selected.name)} />
      <button
        class="quiet icon danger"
        onclick={removeFile}
        title="delete this file from the output directory"
        aria-label="delete this file from the output directory"
      >
        <Trash2 size={14} />
      </button>
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
      {#if metadata}
        <div class="meta">
          {#if metadata.step_name}<div>
              <span class="muted">step</span>
              {metadata.step_name}
            </div>{/if}
          {#if metadata.model_name}<div>
              <span class="muted">model</span>
              <code>{metadata.model_name}</code>
            </div>{/if}
          {#if seed !== undefined}
            <div>
              <span class="muted">seed</span> <code>{seed}</code>
            </div>
          {/if}
          {#if prompt}
            <div class="prompt">
              <span class="muted">prompt</span>
              <p>{prompt}</p>
            </div>
          {/if}
          {#if negativePrompt}
            <div class="prompt">
              <span class="muted">negative prompt</span>
              <p>{negativePrompt}</p>
            </div>
          {/if}
          {#if sourceJob}
            <div>
              <span class="muted">job</span>
              <a
                href={'#/jobs/' + sourceJob.id}
                title="open the job that produced this file"
              >
                {sourceJob.id}
              </a>
            </div>
          {/if}
          {#if embeddedWorkflow}
            <div><span class="muted">workflow</span> {embeddedWorkflow.id}</div>
          {:else}
            <div class="muted">
              no embedded workflow - enable embed_metadata in the step's result
            </div>
          {/if}
        </div>
      {:else if selected.kind === 'image' && metadataLoading}
        <div class="meta muted">reading metadata…</div>
      {:else if selected.kind === 'image'}
        <div class="meta muted">
          no embedded metadata - enable embed_metadata in the step's result
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.4rem 0.8rem;
    margin-bottom: var(--space-4);
  }
  .filter {
    max-width: 220px;
    margin-left: auto;
  }
  .subfolderpick {
    /* The global select rule is width: 100% - here it must share the row */
    width: auto;
    max-width: 200px;
    font-family: var(--font-mono);
    font-size: var(--t-sm);
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
  .cell img,
  .cell video {
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
    margin-right: 0.35rem;
    padding: 0 0.3rem;
    border-radius: 0.2rem;
    background: var(--chip, rgb(255 255 255 / 0.08));
    color: var(--fg);
    font-weight: 600;
    word-break: keep-all;
  }
  .detail {
    position: sticky;
    bottom: 1rem;
    margin-top: 1rem;
    box-shadow: 0 6px 24px rgb(0 0 0 / 0.35);
  }
  .bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.8rem;
    margin-bottom: 0.7rem;
  }
  /* The path the engine wrote, and what you would type to reference it */
  .selname {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
    overflow-wrap: anywhere;
  }
  .icon {
    display: inline-flex;
    padding: 0.3rem 0.45rem;
  }
  .body {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    flex-wrap: wrap;
  }
  .body img,
  .body video {
    max-width: min(480px, 100%);
    border: 1px solid var(--line);
    border-radius: var(--radius-frame);
  }
  .meta {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    font-size: 0.85rem;
    max-width: 46ch;
  }
  .meta .muted {
    margin-right: 0.4rem;
  }
  .prompt p {
    margin: 0.15rem 0 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
</style>
