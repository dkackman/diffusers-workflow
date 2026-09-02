<script lang="ts">
  import { FolderOpen, ImageOff, Trash2, X } from '@lucide/svelte'
  import { api } from '../api'
  import Empty from '../Empty.svelte'
  import { go } from '../router.svelte'
  import { notify } from '../toast'
  import type { GalleryFile } from '../types'

  // One request page - "load more" (and infinite scroll) fetch further
  // pages of this size rather than one unbounded request, so a folder of
  // thousands of outputs never has to be listed in a single round trip.
  const PAGE_SIZE = 120

  let files = $state<GalleryFile[]>([])
  let total = $state(0)
  let folders = $state<string[]>([])
  let folder = $state<string>('') // '' = "all folders" in the UI
  let offset = $state(0)
  let loading = $state(false)
  let filter = $state('')
  let error = $state('')
  let selected = $state<GalleryFile | null>(null)
  let metadata = $state<Record<string, unknown> | null>(null)
  let sourceJob = $state<{ id: string; status: string } | null>(null)
  let sentinel = $state<HTMLDivElement | null>(null)

  const hasMore = $derived(files.length < total)
  // '' server-side means "no folder filter"; the UI's own '' means "all
  // folders", so the literal output-root folder needs a value that survives
  // the round trip - '(root)' can't collide with a real relative path
  const ROOT_FOLDER = '(root)'

  function load(reset: boolean) {
    if (loading) return
    loading = true
    const nextOffset = reset ? 0 : offset
    const folderQuery =
      folder === '' ? undefined : folder === ROOT_FOLDER ? '' : folder
    api
      .gallery(PAGE_SIZE, nextOffset, folderQuery)
      .then((result) => {
        files = reset ? result.files : [...files, ...result.files]
        total = result.total
        offset = nextOffset + result.files.length
        folders = result.folders
        error = ''
      })
      .catch((e) => (error = e.message))
      .finally(() => (loading = false))
  }

  // Reload from the top whenever the folder filter changes; runs once more
  // on mount for the initial load
  $effect(() => {
    void folder
    load(true)
  })

  $effect(() => {
    if (!sentinel) return
    const observer = new IntersectionObserver((entries) => {
      if (entries[0]?.isIntersecting && hasMore && !loading) load(false)
    })
    observer.observe(sentinel)
    return () => observer.disconnect()
  })

  const visible = $derived(
    files.filter((f) => f.name.toLowerCase().includes(filter.toLowerCase())),
  )

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
      total -= 1
      offset -= 1
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
      selected = null
    }
  }}
/>

<div class="head">
  <h1>Gallery</h1>
  <span class="muted">{files.length} of {total} loaded</span>
  {#if folders.length > 1}
    <select class="folder" bind:value={folder} title="filter by source workflow folder">
      <option value="">all folders</option>
      {#each folders as f (f)}
        <option value={f === '' ? '(root)' : f}
          >{f === '' ? '(root)' : f}</option
        >
      {/each}
    </select>
  {/if}
  <input class="filter" placeholder="filter loaded…" bind:value={filter} />
</div>

{#if error}<p class="muted">Could not read the gallery: {error}</p>{/if}
{#if !error && files.length === 0 && !loading}
  <Empty>
    {#snippet icon()}<ImageOff size={36} strokeWidth={1.5} />{/snippet}
    Nothing generated yet — outputs land here as workflows run.
  </Empty>
{/if}

<div class="grid">
  {#each visible as file (file.name)}
    <button
      class="cell"
      class:active={selected?.name === file.name}
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
      <span class="caption" title={file.folder ? `${file.folder}/${file.label}` : file.label}
        >{file.folder ? `${file.folder}/` : ''}{file.label}</span
      >
    </button>
  {/each}
</div>

{#if hasMore}
  <div bind:this={sentinel} class="sentinel">
    <button class="quiet" onclick={() => load(false)} disabled={loading}>
      {loading ? 'loading…' : 'load more'}
    </button>
  </div>
{/if}

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
  .folder {
    margin-left: auto;
  }
  .filter {
    max-width: 220px;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 0.6rem;
  }
  .cell {
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
  .sentinel {
    display: flex;
    justify-content: center;
    padding: 1.2rem 0;
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
