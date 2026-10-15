<script lang="ts">
  // Two to four outputs side by side - the seed sweep, the int4 vs int8
  // pair, the 12 vs 30 steps - with the arguments that differ called out
  // and every video on one transport, so a difference in motion is seen at
  // the same frame rather than remembered across two players
  import { Pause, Play } from '@lucide/svelte'
  import { api } from '../api'
  import { differingArguments, argumentOf, valueText } from '../compare'
  import { formatDuration } from '../format'
  import { wsHref } from '../routes'
  import type { GalleryFile } from '../types'
  import { workspace } from '../workspace.svelte'
  import Markdown from '../ui/Markdown.svelte'
  import Skeleton from '../ui/Skeleton.svelte'

  let { names }: { names: string[] } = $props()

  type Cell = {
    name: string
    file: GalleryFile | null
    meta: Record<string, unknown> | null
    took: number | null
  }
  let cells = $state<Cell[]>([])
  let loaded = $state(false)

  // How long the run took, from the job that made the file - unknown when
  // the job is gone or never finished
  async function runTime(jobId: string | null | undefined) {
    const job = jobId ? await api.getJob(jobId).catch(() => null) : null
    return job?.started_at && job.finished_at
      ? job.finished_at - job.started_at
      : null
  }

  async function cellFor(
    name: string,
    file: GalleryFile | null,
  ): Promise<Cell> {
    if (!file) return { name, file, meta: null, took: null }
    const r = await api.galleryMetadata(name).catch(() => null)
    return {
      name,
      file,
      meta: (r?.metadata as Record<string, unknown> | undefined) ?? null,
      took: await runTime(r?.job?.id),
    }
  }

  async function load(wanted: string[]) {
    length = 0
    time = 0
    playing = false
    const { files } = await api.gallery()
    const byName = new Map(files.map((f) => [f.name, f]))
    cells = await Promise.all(
      wanted.map((name) => cellFor(name, byName.get(name) ?? null)),
    )
  }

  $effect(() => {
    const wanted = [...names]
    loaded = false
    void load(wanted)
      .catch(() => (cells = []))
      .finally(() => (loaded = true))
  })

  const differing = $derived(differingArguments(cells.map((c) => c.meta)))

  // One transport for every video
  let videos = $state<HTMLVideoElement[]>([])
  let playing = $state(false)
  let time = $state(0)
  let length = $state(0)
  // The first video is the clock the slider reads, whatever cells precede it
  const clock = $derived(cells.findIndex((c) => c.file?.kind === 'video'))
  const present = () => videos.filter(Boolean)
  function playPause() {
    playing = !playing
    for (const v of present()) {
      if (!playing) v.pause()
      else void Promise.resolve(v.play()).catch(() => (playing = false))
    }
  }
  function seek(to: number) {
    time = to
    for (const v of present()) v.currentTime = to
  }
</script>

<div class="pagehead baseline">
  <h1>Compare</h1>
  <a class="plain muted" href={wsHref(workspace.current, 'gallery')}
    >back to the gallery</a
  >
</div>

{#if !loaded}
  <Skeleton tiles={names.length} label="loading the comparison" />
{:else}
  {#if cells.some((c) => c.file?.kind === 'video')}
    <div class="transport">
      <button
        class="quiet icon"
        onclick={playPause}
        aria-label={playing ? 'pause all' : 'play all'}
      >
        {#if playing}<Pause size={14} />{:else}<Play size={14} />{/if}
      </button>
      <input
        type="range"
        min="0"
        max={length || 0}
        step="0.01"
        value={time}
        oninput={(e) => seek(Number(e.currentTarget.value))}
        aria-label="position in every video"
      />
      <span class="num muted">{time.toFixed(2)} s</span>
    </div>
  {/if}

  <div class="grid" style:--cols={cells.length}>
    {#each cells as cell, i (cell.name)}
      <figure class="cell">
        {#if !cell.file}
          <p class="muted">{cell.name} is no longer in the gallery.</p>
        {:else if cell.file.kind === 'video'}
          <video
            bind:this={videos[i]}
            src={cell.file.url}
            loop
            muted
            playsinline
            ontimeupdate={(e) => {
              if (i === clock) time = e.currentTarget.currentTime
            }}
            onloadedmetadata={(e) => {
              length = Math.max(length, e.currentTarget.duration)
            }}
          ></video>
        {:else if cell.file.kind === 'image'}
          <img src={cell.file.url} alt={cell.name} />
        {:else}
          <p class="muted">{cell.file.label}</p>
        {/if}
        {#if cell.file}
          <figcaption>
            <span class="mono">{cell.file.label}</span>
            {#if cell.file.version}<span class="num muted"
                >v{cell.file.version}</span
              >{/if}
            <span class="num muted">
              {valueText(argumentOf(cell.meta, 'model'))} · seed {valueText(
                argumentOf(cell.meta, 'seed'),
              )}{cell.took !== null ? ` · ${formatDuration(cell.took)}` : ''}
            </span>
            {#if cell.file.note}<div class="note">
                <Markdown source={cell.file.note} />
              </div>{/if}
          </figcaption>
        {/if}
      </figure>
    {/each}
  </div>

  {#if differing.length}
    <section class="panel">
      <h2>What differs</h2>
      <table>
        <tbody>
          {#each differing as key (key)}
            <tr>
              <th scope="row">{key}</th>
              {#each cells as cell (cell.name)}
                <td class="mono">{valueText(argumentOf(cell.meta, key))}</td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </section>
  {:else if cells.filter((c) => c.meta).length > 1}
    <p class="muted">
      Same model, seed and arguments - the files differ only in what the run
      did.
    </p>
  {/if}
{/if}

<style>
  .grid {
    display: grid;
    grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
    gap: var(--space-3);
    margin-bottom: var(--space-4);
  }
  @media (max-width: 640px) {
    .grid {
      grid-template-columns: 1fr;
    }
  }
  .cell {
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .cell img,
  .cell video {
    width: 100%;
    background: var(--panel-2);
    border: 1px solid var(--line);
  }
  figcaption {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    font-size: var(--t-sm);
  }
  .mono {
    font-family: var(--font-mono);
  }
  .note {
    flex-basis: 100%;
  }
  .transport {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: var(--space-3);
  }
  .transport input {
    flex: 1;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    font-size: var(--t-sm);
  }
  th {
    text-align: left;
    font-family: var(--font-mono);
    color: var(--muted);
    padding-right: var(--space-3);
    white-space: nowrap;
  }
  td {
    padding: var(--space-1) var(--space-2);
    border-top: 1px solid var(--line);
    overflow-wrap: anywhere;
  }
</style>
