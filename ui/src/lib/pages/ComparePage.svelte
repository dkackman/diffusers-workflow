<script lang="ts">
  // Two to four outputs side by side - the seed sweep, the int4 vs int8
  // pair, the 12 vs 30 steps - with the arguments that differ called out
  // and every video on one transport, so a difference in motion is seen at
  // the same frame rather than remembered across two players
  import { Columns2, Pause, Play, X } from '@lucide/svelte'
  import { api } from '../api'
  import { untrack } from 'svelte'
  import { hasPicture, outputMedia, viewLabels, withoutRun } from '../media'
  import { openInHash } from '../mediaViewer'
  import { hashQuery, replaceRoute } from '../router.svelte'
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
  import { SECTION_LABELS, backFrom, compareHref, wsHref } from '../routes'
  import { workspace } from '../workspace.svelte'
  import Markdown from '../ui/Markdown.svelte'
  import MediaBody from '../ui/MediaBody.svelte'
  import MediaViewer from '../ui/MediaViewer.svelte'
  import MissingFile from '../ui/MissingFile.svelte'
  import { missingFile } from '../missingFile.svelte'
  import Skeleton from '../ui/Skeleton.svelte'

  let { names }: { names: string[] } = $props()
  // The router hands a new array on every hashchange, the viewer's `?open=`
  // among them: what is compared changes only when the names themselves do
  const key = $derived(names.join('\n'))

  // A file's note or a new run shows within one feed interval. A cell that
  // already had its file keeps the metadata and run time it has - only a
  // cell that was missing its file, or is new, asks for them. `key` is read
  // before the first await, so new names reload loudly
  const comparison: Resource<Cell[]> = resource(
    async () => {
      void key
      const wanted = untrack(() => [...names])
      // Under two there is nothing to compare, so nothing is read
      if (wanted.length < 2) return []
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
  // New names start the transport over, the clips with it
  $effect(() => {
    void key
    length = 0
    untrack(() => {
      for (const v of present()) v.pause()
      seek(0)
      playing = false
    })
  })

  const differing = $derived(differingArguments(cells.map((c) => c.meta)))

  // One transport for every video, kept by cell name
  let videos: Record<string, HTMLVideoElement> = {}
  let playing = $state(false)
  let time = $state(0)
  let length = $state(0)
  // The first video is the clock the slider reads, whatever cells precede it
  const clock = $derived(cells.find((c) => c.file?.kind === 'video')?.name)
  const present = () => Object.values(videos)
  // A quiet reload hands the same element over again: listen once per element
  const listening = new WeakSet<HTMLVideoElement>()
  function track(name: string, el: HTMLVideoElement | null) {
    if (!el) {
      delete videos[name]
      return
    }
    videos[name] = el
    const measure = () => (length = Math.max(length, el.duration))
    if (!listening.has(el)) {
      listening.add(el)
      el.addEventListener('timeupdate', () => {
        if (name === clock) time = el.currentTime
      })
      el.addEventListener('loadedmetadata', measure)
    }
    if (el.readyState >= 1) measure()
  }
  function playPause() {
    playing = !playing
    for (const v of present()) {
      if (!playing) v.pause()
      else void Promise.resolve(v.play()).catch(() => (playing = false))
    }
  }
  const galleryHref = $derived(wsHref(workspace.current, 'gallery'))
  // Where the comparison was opened from (`back=` in the hash), else the Gallery
  const from = $derived.by(() => {
    void names.join('\n')
    return backFrom(hashQuery().back)
  })
  const back = $derived(
    from ?? { href: galleryHref, label: SECTION_LABELS.ws.gallery },
  )
  function seek(to: number) {
    time = to
    for (const v of present()) v.currentTime = to
  }

  // The viewer steps through the pictures and clips compared, as the
  // Gallery's does: the open file is `open=` in the hash (mediaViewer.ts)
  // Two same-leaf files compared are told apart as the Gallery tells them
  const labelled = $derived.by(() => {
    const files = cells.flatMap((c) => (c.file ? [c.file] : []))
    return viewLabels(
      files.map((f) => f.name),
      withoutRun(files),
    )
  })
  const shown = (cell: Cell) =>
    cell.file ? labelled(outputMedia(cell.file, workspace.current)) : null
  const items = $derived(
    cells.flatMap((c) => {
      const tile = shown(c)
      return tile ? [tile] : []
    }),
  )
  let open = $state<string | null>(openInHash())
  // The viewer plays its own file: the synced clips stop under it
  $effect(() => {
    if (open === null || !playing) return
    playing = false
    for (const v of untrack(present)) v.pause()
  })
  // An open= naming no file compared is said and cleared (missingFile.svelte.ts)
  const stray = missingFile({
    open: () => open,
    known: () => comparison.status === 'ready',
    names: () => items.map((item) => item.name),
  })
  $effect(() => void (stray.missing && untrack(() => (open = null))))

  const without = (name: string) =>
    replaceRoute(
      compareHref(
        workspace.current,
        names.filter((n) => n !== name),
        from ?? undefined,
      ),
    )
</script>

<PageHeader title="Compare" baseline {back} />

<MissingFile leaf={stray.missing} ondismiss={stray.dismiss} />

<LoadState of={comparison} label="the comparison" empty={names.length < 2}>
  {#snippet emptyIcon()}<Columns2 size={36} strokeWidth={1.5} />{/snippet}
  {#snippet emptyText()}<span>Nothing to compare yet.</span><br /><span>
      Tick two or more files in the Gallery, then press Compare.
    </span><br /><a href={galleryHref}>Open the Gallery</a>{/snippet}
  {#snippet skeleton()}{#if names.length >= 2}<Skeleton
        tiles={names.length}
        label="loading the comparison"
      />{/if}{/snippet}
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
    {#each cells as cell (cell.name)}
      {@const tile = shown(cell)}
      <figure class="cell">
        {#if !tile}
          <p class="muted">{cell.name} is no longer in the gallery.</p>
        {:else if hasPicture(tile.kind)}
          <button
            type="button"
            class="thumb"
            aria-label="Open {tile.label}"
            title="open this file large"
            onclick={() => (open = cell.name)}
          >
            <MediaBody
              file={tile}
              fit={false}
              controls={false}
              muted
              onvideo={(el) => track(cell.name, el)}
            />
          </button>
        {:else}
          <div class="sound">
            <MediaBody file={tile} fit={false} />
            <button
              type="button"
              class="quiet"
              aria-label="Open {tile.label}"
              title="open this file large"
              onclick={() => (open = cell.name)}
            >
              Open
            </button>
          </div>
        {/if}
        <button
          type="button"
          class="quiet icon remove"
          aria-label="Remove from comparison"
          title="Remove from comparison"
          onclick={() => without(cell.name)}
        >
          <X size={14} />
        </button>
        {#if cell.file}
          <figcaption>
            <span class="mono">{tile?.label}</span>
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
      <table class="datatable">
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

<MediaViewer {items} bind:open ongone={stray.gone} />

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
  .cell {
    position: relative;
  }
  .remove {
    position: absolute;
    top: var(--space-1);
    right: var(--space-1);
  }
  .sound {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    align-items: flex-start;
  }
  .thumb {
    display: block;
    padding: 0;
    border: 0;
    background: none;
    cursor: zoom-in;
  }
  .cell :global(.media-body) {
    display: block;
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
    font-size: var(--t-sm);
  }
  th {
    font-family: var(--font-mono);
    font-size: inherit;
    white-space: nowrap;
  }
  td {
    overflow-wrap: anywhere;
  }
</style>
