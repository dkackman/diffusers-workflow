<script lang="ts">
  // Two to four outputs side by side - the seed sweep, the int4 vs int8
  // pair, the 12 vs 30 steps - with the arguments that differ called out
  // and every video on one transport, so a difference in motion is seen at
  // the same frame rather than remembered across two players
  import { Columns2, Pause, Play } from '@lucide/svelte'
  import { api } from '../api'
  import { untrack } from 'svelte'
  import { resource, type Resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import {
    argumentOf,
    cellFor,
    differingArguments,
    valueText,
    type Cell,
  } from '../compare'
  import { formatDuration } from '../format'
  import { versionTitle } from '../terms'
  import { SECTION_LABELS, wsHref } from '../routes'
  import { workspace } from '../workspace.svelte'
  import Markdown from '../ui/Markdown.svelte'
  import Skeleton from '../ui/Skeleton.svelte'

  let { names }: { names: string[] } = $props()

  // A file's note or a new run shows within one feed interval. A cell that
  // already had its file keeps the metadata and run time it has - only a
  // cell that was missing its file, or is new, asks for them. `names` is read
  // before the first await, so new names reload loudly
  const comparison: Resource<Cell[]> = resource(
    async () => {
      const wanted = [...names]
      const held = new Map(
        untrack(() => comparison.data ?? []).map((c) => [c.name, c]),
      )
      const { files } = await api.gallery()
      const byName = new Map(files.map((f) => [f.name, f]))
      return Promise.all(
        wanted.map((name) => {
          const file = byName.get(name) ?? null
          const had = held.get(name)
          return had?.file && had.meta && file
            ? Promise.resolve({ ...had, file })
            : cellFor(name, file)
        }),
      )
    },
    { watch: ['gallery'] },
  )
  const cells = $derived(comparison.data ?? [])
  // New names start the transport over
  $effect(() => {
    void names.join('\n')
    length = 0
    time = 0
    playing = false
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
  const galleryHref = $derived(wsHref(workspace.current, 'gallery'))
  function seek(to: number) {
    time = to
    for (const v of present()) v.currentTime = to
  }
</script>

<PageHeader
  title="Compare"
  baseline
  back={{ href: galleryHref, label: SECTION_LABELS.ws.gallery }}
/>

<LoadState of={comparison} label="the comparison" empty={names.length < 2}>
  {#snippet emptyIcon()}<Columns2 size={36} strokeWidth={1.5} />{/snippet}
  {#snippet emptyText()}<span>Nothing to compare yet.</span><br /><span>
      Tick two or more files in the Gallery, then press Compare.
    </span><br /><a href={galleryHref}>Open the Gallery</a>{/snippet}
  {#snippet skeleton()}<Skeleton
      tiles={names.length}
      label="loading the comparison"
    />{/snippet}
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
            {#if cell.file.version}<span
                class="num muted"
                title={versionTitle(cell.file.version)}
              >
                v{cell.file.version}
              </span>{/if}
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
      Same model, seed and arguments - the files differ only in what the job
      did.
    </p>
  {/if}
</LoadState>

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
