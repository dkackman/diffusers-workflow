<script lang="ts">
  /* A chained step's segments, one row each, read-only (#1475): the
   * segment's frame, how many renders its check tried, each measure its
   * kept attempt computed against the ceiling, and whether it was resumed or
   * repaired. One panel per chained step in the job's manifest; a job with
   * no chained step renders nothing. The rows are `chainRows`'s.
   *
   * Each row offers Repair this segment, and a take that stopped short of
   * its plan offers Continue the chain (#1476): when each is offered, and
   * what it sends, is `chainActions`'s. Repair opens the repair template's
   * workflow page with its fields filled; nothing runs until Run there.
   * Continue reruns this job, behind a confirm. */
  import { api, outputUrl } from '../api'
  import {
    REPAIR_TEMPLATE,
    WORKFLOW_PREFILL_KEY,
    chainModel,
    continueAction,
    repairAction,
    repairModels,
    type ContinueAction,
  } from '../chainActions'
  import { confirmDialog } from '../confirm.svelte'
  import { go } from '../router.svelte'
  import { notify } from '../toast'
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

  // The models the repair template loads: read once there is a chain to
  // offer it on. Unreadable, it repairs nothing
  let models = $state<string[]>([])
  const hasChain = $derived(tables.length > 0)
  $effect(() => {
    if (!hasChain) return
    let stopped = false
    try {
      api
        .getWorkflow(REPAIR_TEMPLATE)
        .then(({ definition: repair }) => {
          if (!stopped) models = repairModels(repair as Record<string, any>)
        })
        .catch(() => {
          /* no repair template on this server */
        })
    } catch {
      /* no repair template on this server */
    }
    return () => {
      stopped = true
    }
  })

  const args = $derived(job.arguments as Record<string, unknown>)
  const repairFor = (table: ChainTable, row: ChainRow) =>
    repairAction(table, row, chainModel(definition, table.step, args), models)

  function repair(table: ChainTable, row: ChainRow) {
    const action = repairFor(table, row)
    if (!action.enabled) return
    sessionStorage.setItem(
      WORKFLOW_PREFILL_KEY,
      JSON.stringify({
        workflow: action.workflow,
        arguments: action.arguments,
      }),
    )
    go('ws', job.workspace, 'workflows', ...action.workflow.split('/'))
  }

  let continuing = $state(false)
  async function continueChain(table: ChainTable, action: ContinueAction) {
    const remaining = (table.planned ?? 0) - table.rows.length
    const ok = await confirmDialog(
      `Queue this job again from its last kept segment (${table.rows.length} of ${table.planned})? ` +
        `It renders the remaining ${remaining} segment${remaining === 1 ? '' : 's'} of ${table.step}.`,
      { confirmLabel: 'Continue' },
    )
    if (!ok) return
    continuing = true
    try {
      const next = await api.rerunJob(job.id, { arguments: action.arguments })
      go('ws', job.workspace, 'jobs', next.id)
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
    } finally {
      continuing = false
    }
  }

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
  {@const more = continueAction(table, definition)}
  <section
    class="panel chain-segments"
    aria-label="Chain segments: {table.step}"
  >
    <h2>
      Segments <span class="step mono">{table.step}</span>
      <span class="planned">{plannedText(table)}</span>
    </h2>
    {#if more}
      <div class="continue">
        <button
          class="quiet"
          disabled={continuing}
          onclick={() => continueChain(table, more)}
          title="queue this job again, resuming from segment {table.rows
            .length} and rendering the rest of the plan"
          >{continuing ? 'Queueing…' : 'Continue the chain'}</button
        >
      </div>
    {/if}
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
            <th scope="col"><span class="sr-only">Actions</span></th>
          </tr>
        </thead>
        <tbody>
          {#each table.rows as row (row.index)}
            {@const action = repairFor(table, row)}
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
              <td class="repair">
                <button
                  class="quiet"
                  disabled={!action.enabled}
                  title={action.enabled
                    ? `open ${action.workflow} with this segment filled in - nothing runs until you press Run`
                    : action.reason}
                  onclick={() => repair(table, row)}>Repair this segment</button
                >
                {#if !action.enabled}<span class="reason muted"
                    >{action.reason}</span
                  >{/if}
              </td>
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
  .continue {
    margin-bottom: 0.5rem;
  }
  .repair button {
    white-space: nowrap;
    font-size: 0.8rem;
  }
  .repair .reason {
    display: block;
    max-width: 16rem;
    font-size: 0.75rem;
  }
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip: rect(0 0 0 0);
    white-space: nowrap;
  }
</style>
