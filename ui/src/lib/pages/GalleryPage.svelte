<script lang="ts">
  import { ImageOff, FolderOpen, Trash2, X, Download } from '@lucide/svelte'
  import DownloadLink from '../DownloadLink.svelte'
  import { api } from '../api'
  import Empty from '../Empty.svelte'
  import FolderGroups from '../FolderGroups.svelte'
  import { go } from '../router.svelte'
  import { SvelteSet } from 'svelte/reactivity'
  import { notify } from '../toast'
  import type { GalleryFile } from '../types'

  let files = $state<GalleryFile[]>([])
  let loaded = $state(false)
  let filter = $state('')
  let error = $state('')
  let selected = $state<GalleryFile | null>(null)
  // Names ticked in the grid, for the bulk actions
  const picked = new SvelteSet<string>()
  // Anchor for a shift-click range, in the order the grid renders
  let anchor = $state<string | null>(null)
  let busy = $state(false)
  let metadata = $state<Record<string, unknown> | null>(null)
  let sourceJob = $state<{ id: string; status: string } | null>(null)

  $effect(() => {
    api
      .gallery()
      .then((result) => {
        files = result.files
        error = ''
        prune()
      })
      .catch((e) => (error = e.message))
      .finally(() => (loaded = true))
  })

  const visible = $derived(
    files.filter((f) => f.name.toLowerCase().includes(filter.toLowerCase())),
  )
  const byName = $derived(new Map(visible.map((f) => [f.name, f])))

  const pickedCount = $derived(picked.size)
  /** The selection in the order the grid shows them, which is the order
   * the bulk actions act in and the order a shift-range spans. */
  const pickedNames = $derived(
    visible.filter((f) => picked.has(f.name)).map((f) => f.name),
  )

  /** Forget names no longer in the listing, so a file deleted elsewhere
   * cannot linger in the selection and fail every later bulk action. */
  function prune() {
    const present = new Set(files.map((f) => f.name))
    for (const name of [...picked]) if (!present.has(name)) picked.delete(name)
  }

  function togglePick(name: string, shift: boolean) {
    if (shift && anchor !== null) {
      const order = visible.map((f) => f.name)
      const from = order.indexOf(anchor)
      const to = order.indexOf(name)
      if (from !== -1 && to !== -1) {
        const [low, high] = from < to ? [from, to] : [to, from]
        // A range always selects: extending a selection is what shift is
        // for, and toggling each cell would make the result depend on
        // whatever the range happened to contain
        for (const each of order.slice(low, high + 1)) picked.add(each)
        anchor = name
        return
      }
    }
    if (picked.has(name)) picked.delete(name)
    else picked.add(name)
    anchor = name
  }

  function selectAllVisible() {
    for (const file of visible) picked.add(file.name)
  }

  function clearPicked() {
    picked.clear()
    anchor = null
  }

  async function downloadPicked() {
    busy = true
    try {
      await api.archiveOutputs(pickedNames)
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
    } finally {
      busy = false
    }
  }

  async function removePicked() {
    const names = pickedNames
    if (
      !window.confirm(
        `Delete ${names.length} file${names.length === 1 ? '' : 's'}? This removes them on disk.`,
      )
    )
      return
    busy = true
    // Sequential rather than parallel: a selection of hundreds should not
    // open hundreds of sockets, and the order makes the log readable
    const failed: string[] = []
    for (const name of names) {
      try {
        await api.deleteOutput(name)
      } catch {
        failed.push(name)
      }
    }
    const gone = new Set(names.filter((name) => !failed.includes(name)))
    files = files.filter((f) => !gone.has(f.name))
    // Whatever could not be deleted stays selected, so a retry needs no
    // re-ticking and the failure is visible rather than silently dropped
    picked.clear()
    for (const name of failed) picked.add(name)
    anchor = null
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
    sourceJob = null
    api.galleryMetadata(file.name).then((r) => {
      if (selected?.name === file.name) {
        metadata = r.metadata
        sourceJob = r.job
      }
    })
  }

  async function removeFile() {
    if (!selected) return
    if (
      !window.confirm(`Delete ${selected.name}? This removes the file on disk.`)
    )
      return
    const name = selected.name
    try {
      await api.deleteOutput(name)
      files = files.filter((f) => f.name !== name)
      // A file deleted from here may also be ticked in the grid; leaving it
      // there would make the next bulk action fail on a file that is gone
      picked.delete(name)
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
    go('edit')
  }

  const day = (mtime: number) => new Date(mtime * 1000).toLocaleString()
  const mb = (size: number) => (size / (1024 * 1024)).toFixed(1) + ' MB'
</script>

<svelte:window
  onkeydown={(e) => {
    if (e.key === 'Escape') {
      if (document.querySelector('[role="dialog"]')) return
      // The selection is the more recent, more surprising state to be
      // stuck in, so it clears first and the detail panel on a second press
      if (picked.size) clearPicked()
      else selected = null
    }
  }}
/>

<div class="head">
  <h1>Gallery</h1>
  <span class="muted">{files.length} files</span>
  <input class="filter" placeholder="filter…" bind:value={filter} />
</div>

{#if error}<p class="muted">Could not read the gallery: {error}</p>{/if}
{#if loaded && !error && files.length === 0}
  <Empty>
    {#snippet icon()}<ImageOff size={36} strokeWidth={1.5} />{/snippet}
    Nothing generated yet — outputs land here as workflows run.
  </Empty>
{/if}

{#if pickedCount}
  <div class="picks panel" role="region" aria-label="selected files">
    <strong>{pickedCount} selected</strong>
    <span class="flex"></span>
    <button class="withicon" onclick={downloadPicked} disabled={busy}>
      <Download size={14} />Download .zip
    </button>
    <button class="withicon danger" onclick={removePicked} disabled={busy}>
      <Trash2 size={14} />Delete
    </button>
    <button class="quiet" onclick={clearPicked} disabled={busy}>Clear</button>
  </div>
{/if}

<div class="picktools">
  <button
    class="quiet"
    onclick={selectAllVisible}
    disabled={visible.length === 0}
    >Select all{filter ? ' matching' : ''} ({visible.length})</button
  >
</div>

<FolderGroups
  names={visible.map((f) => f.name)}
  collapseKey="collapsed-gallery-folders"
  filterActive={filter !== ''}
  minColumn="150px"
>
  {#snippet card(name)}
    {@const file = byName.get(name)!}
    <div class="cellwrap" class:picked={picked.has(name)}>
      <!-- A sibling of the cell rather than a child: a checkbox nested in
           a button is invalid, and keeping them apart is what lets a plain
           click still open the details it always has -->
      <input
        class="pick"
        type="checkbox"
        checked={picked.has(name)}
        aria-label="select {name}"
        title="select this file for a bulk action"
        onclick={(e) => togglePick(name, e.shiftKey)}
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
        <span class="caption" title={file.name}>{file.label}</span>
      </button>
    </div>
  {/snippet}
</FolderGroups>

{#if selected}
  <div class="detail panel">
    <div class="bar">
      <strong>{selected.name}</strong>
      <span class="muted">{mb(selected.size)} · {day(selected.mtime)}</span>
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
      <a
        href={selected.url}
        target="_blank"
        class="muted"
        title="open the file itself in a new tab">open file</a
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
              {metadata.model_name}
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
      {:else if selected.kind === 'image'}
        <div class="meta muted">reading metadata…</div>
      {/if}
    </div>
  </div>
{/if}

<style>
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.4rem 1rem;
    margin-bottom: 1rem;
  }
  .filter {
    max-width: 220px;
    margin-left: auto;
  }
  .picks {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.8rem;
    margin-bottom: 0.6rem;
    position: sticky;
    top: 0;
    z-index: 2;
  }
  .picktools {
    margin-bottom: 0.6rem;
  }
  .cellwrap {
    position: relative;
    display: flex;
  }
  .pick {
    position: absolute;
    top: 0.55rem;
    left: 0.55rem;
    z-index: 1;
    margin: 0;
    cursor: pointer;
    /* Out of the way until it is wanted: hovering the tile, focusing the
       box itself, or any selection existing at all brings it back */
    opacity: 0;
  }
  .cellwrap:hover .pick,
  .pick:focus-visible,
  .pick:checked {
    opacity: 1;
  }
  .cellwrap.picked .cell {
    border-color: var(--accent);
  }
  .cell {
    flex: 1;
    min-width: 0;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.4rem;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    color: var(--muted);
    font-weight: 500;
    font-size: 0.75rem;
    /* A folder of thousands of outputs still renders every cell's DOM node
       (no true windowing library is in the project's dependencies yet),
       but content-visibility skips layout/paint for cells scrolled out of
       view, which is most of the win for cheap. contain-intrinsic-size
       keeps scrollbar height stable before a cell has ever been measured. */
    content-visibility: auto;
    contain-intrinsic-size: 150px 190px;
  }
  .cell:hover,
  .cell.active {
    border-color: var(--accent);
    filter: none;
  }
  .cell img,
  .cell video {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    border-radius: 5px;
    display: block;
  }
  .audio {
    aspect-ratio: 1;
    display: grid;
    place-items: center;
  }
  .caption {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
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
  .flex {
    flex: 1;
  }
  .withicon {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
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
    border-radius: 6px;
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
