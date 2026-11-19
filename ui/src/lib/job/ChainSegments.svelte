<script lang="ts">
  /* A chained step's segments, one row each, read-only (#1475): the
   * segment's frame, how many renders its check tried, each measure its
   * kept attempt computed against the ceiling, and whether it was resumed or
   * repaired. One panel per chained step in the job's manifest; a job with
   * no chained step renders nothing. The rows are `chainRows`'s. */
  import { outputUrl } from '../api'
  import {
    attemptsText,
    chainContext,
    chainRows,
    measureText,
    thumbSrc,
    type ChainRow,
    type ChainTable,
  } from '../chainRows'
  import type { JobDetail } from '../types'

  let {
    job,
    definition = null,
  }: { job: JobDetail; definition?: Record<string, any> | null } = $props()

  const tables = $derived(
    (job.manifest ?? [])
      .map((entry) =>
        chainRows(
          entry,
          chainContext(
            definition,
            entry.step,
            job.arguments as Record<string, unknown>,
          ),
        ),
      )
      .filter((table): table is ChainTable => table !== null),
  )

  const fileUrl = (path: string) => outputUrl(path, job.id, job.workspace)

  const plannedText = (table: ChainTable) =>
    table.planned === null
      ? `${table.rows.length} segments`
      : `${table.rows.length} of ${table.planned} planned`

  // Not every row records every measure: a seam measure has no value on
  // segment 1. The table's columns are the measures any row records
  const columns = (table: ChainTable) => table.measureNames

  const anyChecked = (table: ChainTable) =>
    table.rows.some((row) => row.measureState !== 'unchecked')

  // Load a thumbnail only once it is near the viewport: a long chain is
  // dozens of videos. Without an observer (jsdom), load at once
  function lazy(node: HTMLVideoElement, src: string) {
    const load = () => {
      if (node.getAttribute('src') !== src) node.src = src
    }
    if (typeof IntersectionObserver === 'undefined') {
      load()
      return { update: (next: string) => ((src = next), load()) }
    }
    const observer = new IntersectionObserver(
      (seen) => {
        if (seen.some((e) => e.isIntersecting)) {
          load()
          observer.disconnect()
        }
      },
      { rootMargin: '200px' },
    )
    observer.observe(node)
    return {
      update(next: string) {
        src = next
        if (node.getAttribute('src')) load()
      },
      destroy: () => observer.disconnect(),
    }
  }

  // With no frame rate on record, seek by the segment's share of the take
  function seekByFraction(event: Event, row: ChainRow) {
    if (!row.thumb || row.thumb.seconds !== null) return
    const video = event.currentTarget as HTMLVideoElement
    if (Number.isFinite(video.duration) && video.duration > 0)
      video.currentTime = video.duration * row.thumb.fraction
  }

  const measureLabel = (row: ChainRow) =>
    row.measureState === 'resumed'
      ? 'not recorded - resumed from an earlier run'
      : row.measureState === 'not-recorded'
        ? 'not recorded'
        : 'nothing measured'
</script>

{#each tables as table (table.step)}
  <section
    class="panel chain-segments"
    aria-label="Chain segments: {table.step}"
  >
    <h2>
      Segments <span class="step mono">{table.step}</span>
      <span class="planned">{plannedText(table)}</span>
    </h2>
    <div class="scroll">
      <table>
        <thead>
          <tr>
            <th scope="col">#</th>
            <th scope="col">Frame</th>
            <th scope="col">Segment</th>
            <th scope="col">Attempts</th>
            {#if columns(table).length}
              {#each columns(table) as name (name)}
                <th scope="col" class="mono">{name}</th>
              {/each}
            {:else if anyChecked(table)}
              <th scope="col">Measures</th>
            {/if}
          </tr>
        </thead>
        <tbody>
          {#each table.rows as row (row.index)}
            <tr data-segment={row.index}>
              <td class="index mono">{row.index}</td>
              <td class="thumb">
                {#if row.thumb}
                  <video
                    use:lazy={thumbSrc(fileUrl(row.thumb.file), row.thumb)}
                    preload="metadata"
                    muted
                    playsinline
                    onloadedmetadata={(e) => seekByFraction(e, row)}
                    title="{row.name} - frame {row.startFrame}"
                  ></video>
                {:else}
                  <span class="muted">no file</span>
                {/if}
              </td>
              <td class="name">
                <span class="mono">{row.name.replace(' (repaired)', '')}</span>
                {#if row.repaired}<span class="mark">(repaired)</span>{/if}
                {#if row.resumed}<span class="mark">resumed</span>{/if}
                <span class="frames mono"
                  >frames {row.startFrame}–{row.startFrame +
                    row.numFrames -
                    1}</span
                >
              </td>
              <td class="attempts">
                <span class="mono">{attemptsText(row)}</span>
                {#if row.keptSeed !== null}
                  <span class="seed"
                    >kept seed <span class="mono">{row.keptSeed}</span></span
                  >
                {/if}
              </td>
              {#if row.measureState === 'recorded' && Object.keys(row.measures).length}
                {#each columns(table) as name (name)}
                  {@const cell = row.measures[name]}
                  <td class="measure {cell?.verdict ?? ''}">
                    {#if cell}
                      <span class="mono">{measureText(cell)}</span>
                      <span class="verdict">{cell.verdict}</span>
                    {:else}
                      <span class="muted">-</span>
                    {/if}
                  </td>
                {/each}
              {:else if row.measureState !== 'unchecked' && (columns(table).length || anyChecked(table))}
                <td
                  class="muted not-recorded"
                  colspan={Math.max(columns(table).length, 1)}
                  >{measureLabel(row)}</td
                >
              {:else if columns(table).length || anyChecked(table)}
                <td colspan={Math.max(columns(table).length, 1)}></td>
              {/if}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>
{/each}

<style>
  h2 {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.5rem;
  }
  .step {
    font-size: 0.85em;
    font-weight: normal;
  }
  .planned {
    margin-left: auto;
    font-size: 0.85rem;
    font-weight: normal;
    color: var(--muted);
  }
  .mono {
    font-family: var(--font-mono);
  }
  .scroll {
    overflow-x: auto;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.85rem;
  }
  th,
  td {
    text-align: left;
    vertical-align: top;
    padding: 0.35rem 0.5rem;
    border-bottom: 1px solid var(--line);
  }
  th {
    font-weight: 600;
    color: var(--muted);
    white-space: nowrap;
  }
  .thumb video {
    display: block;
    width: 112px;
    aspect-ratio: 16 / 9;
    object-fit: cover;
    background: var(--panel-2);
    border-radius: 3px;
  }
  .name,
  .attempts {
    white-space: nowrap;
  }
  .name > span,
  .attempts > span {
    display: block;
  }
  .frames,
  .seed {
    color: var(--muted);
    font-size: 0.8rem;
  }
  .mark {
    font-size: 0.8rem;
    color: var(--muted);
    font-style: italic;
  }
  .measure {
    white-space: nowrap;
  }
  .measure .verdict {
    display: block;
    font-size: 0.8rem;
    font-weight: 600;
  }
  .measure.pass .verdict {
    color: var(--good);
  }
  .measure.warn .verdict {
    color: var(--warn);
  }
  .measure.fail .verdict {
    color: var(--bad);
  }
  .measure.fail {
    box-shadow: inset 3px 0 0 var(--bad);
  }
  .measure.warn {
    box-shadow: inset 3px 0 0 var(--warn);
  }
  .muted {
    color: var(--muted);
  }
</style>
