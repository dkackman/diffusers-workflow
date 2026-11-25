<script lang="ts">
  // One job in the Jobs list: its picture, workflow, prompt and note, the
  // chips that say where it stands, and under them its id, which never
  // truncates - it is what tells two runs of one workflow apart. The same
  // cover slot and "<time> · <id>" line as the Overview's job rows.
  import { ChevronDown, ChevronUp, ChevronsUp } from '@lucide/svelte'
  import { formatDuration, formatExact } from '../format'
  import Time from '../ui/Time.svelte'
  import { holdKind, holdText } from '../progress'
  import { queueWait } from '../jobTimes'
  import { firstLine } from '../jobsListing'
  import { wsHref } from '../routes'
  import { versionTitle, workflowName } from '../terms'
  import type { GalleryFile, JobSummary } from '../types'
  import StatusChip from '../ui/StatusChip.svelte'
  import ItemProof from '../request/ItemProof.svelte'

  let {
    job,
    queueSize,
    showWorkspace,
    prompt,
    percent,
    proof,
    onmove,
    busyId = null,
  }: {
    job: JobSummary
    queueSize: number
    showWorkspace: boolean
    /** What the job was asked for; '' when it carried no prompt */
    prompt: string
    /** How far a running job's denoise loop is, in whole percent */
    percent: number | null
    /** The run's proof, listed in this job's workspace */
    proof: GalleryFile | null
    onmove: (e: MouseEvent, id: string, way: 'up' | 'down' | 'front') => void
    /** The job whose move is in flight: it is busy, and no move starts
     * until it ends (a click is dropped), so every move button says so */
    busyId?: string | null
  } = $props()

  const requestLabel = $derived(
    job.item ? `${job.request} › ${job.item}` : (job.request ?? ''),
  )
  const wait = $derived(queueWait(job, Date.now() / 1000))
  const duration = $derived(
    job.started_at
      ? formatDuration((job.finished_at ?? Date.now() / 1000) - job.started_at)
      : '',
  )
  // The row shows the compact start; hovering gives it to the second, with
  // the finish and any time spent queued
  const times = $derived(
    [
      job.started_at && `started ${formatExact(job.started_at)}`,
      job.finished_at && `finished ${formatExact(job.finished_at)}`,
      wait !== null &&
        wait >= 1 &&
        `waited ${formatDuration(wait)} in the queue`,
    ]
      .filter(Boolean)
      .join('\n'),
  )
  const promptLine = $derived(firstLine(prompt))
  const noteLine = $derived(firstLine(job.note ?? ''))
  const hold = $derived(job.waiting_for ? holdKind(job.waiting_for) : null)
</script>

<a
  class="row plain"
  class:historical={job.historical}
  class:row-live={job.status === 'running'}
  href={wsHref(job.workspace, 'jobs', job.id)}
  title={job.historical
    ? 'finished before this server started - loaded from history'
    : ''}
>
  <span class="proof">
    <ItemProof
      output={proof ? { name: proof.name, kind: proof.kind } : null}
      state={job.status}
      workspace={job.workspace}
      alt=""
      size="cover"
    />
  </span>
  <span class="state">
    <StatusChip status={job.status} />
    {#if percent !== null}
      <span class="pct" title="how far its denoise loop is">{percent}%</span>
    {/if}
  </span>
  <span class="namecol">
    <span class="name">
      {workflowName(job)}
      {#if job.run_version}
        <span class="wschip muted" title={versionTitle(job.run_version)}>
          v{job.run_version}
        </span>
      {/if}
      {#if job.rerun_of}
        <span class="wschip muted" title="a rerun of {job.rerun_of}">
          rerun
        </span>
      {/if}
      {#if job.request}
        <span
          class="wschip muted"
          title="request {job.request}{job.item ? `, item ${job.item}` : ''}"
        >
          {requestLabel}
        </span>
      {/if}
      {#if showWorkspace}
        <span class="wschip muted" title="workspace">{job.workspace}</span>
      {/if}
      {#if job.device}
        <span class="wschip muted" title="the card this job ran on">
          {job.device}
        </span>
      {/if}
      {#if job.acknowledged === 'bound'}
        <span
          class="wschip muted"
          title="queued with a cost acknowledgement bound to the validated plan"
        >
          acknowledged
        </span>
      {/if}
      {#if job.queue_position !== undefined}
        <span class="qpos" title="position in the waiting queue">
          #{job.queue_position + 1}
        </span>
        {#if queueSize > 1}
          <span class="qmove">
            {#if job.queue_position > 0}
              <button
                class="quiet icon"
                onclick={(e) => onmove(e, job.id, 'front')}
                title="run this job next"
                aria-busy={busyId === job.id}
                aria-disabled={busyId !== null}
                aria-label="move {job.id} to the front of the queue"
              >
                <ChevronsUp size={13} />
              </button>
              <button
                class="quiet icon"
                onclick={(e) => onmove(e, job.id, 'up')}
                title="move up one place"
                aria-busy={busyId === job.id}
                aria-disabled={busyId !== null}
                aria-label="move {job.id} up one place"
              >
                <ChevronUp size={13} />
              </button>
            {/if}
            {#if job.queue_position < queueSize - 1}
              <button
                class="quiet icon"
                onclick={(e) => onmove(e, job.id, 'down')}
                title="move down one place"
                aria-busy={busyId === job.id}
                aria-disabled={busyId !== null}
                aria-label="move {job.id} down one place"
              >
                <ChevronDown size={13} />
              </button>
            {/if}
          </span>
        {/if}
      {/if}
    </span>
    {#if promptLine}
      <span class="line prompt muted" title={prompt}>{promptLine}</span>
    {/if}
    {#if noteLine}
      <span class="line note" title={job.note ?? ''}>{noteLine}</span>
    {/if}
    {#if job.reason}
      <span class="line reason muted" title={job.reason}>{job.reason}</span>
    {/if}
    <code class="muted jobid">{job.id}</code>
  </span>
  <span class="timing">
    {#if job.status === 'queued'}
      <span
        class="muted started"
        title={job.created_at
          ? `queued ${formatExact(job.created_at)} - time spent waiting for a card`
          : 'time spent waiting for a card'}
      >
        waiting {wait === null ? '' : formatDuration(wait)}
      </span>
      {#if job.waiting_for}
        <span class="muted" title={holdText(job.waiting_for)}>
          · {hold === 'host_ram'
            ? 'host RAM'
            : hold === 'cards_busy'
              ? 'cards busy'
              : 'held'}
        </span>
      {/if}
    {:else}
      <span class="muted started" title={times}>
        {#if job.started_at}<Time
            at={job.started_at}
            title={times}
          />{:else}—{/if}
      </span>
    {/if}
    <span class="muted dur">{duration}</span>
  </span>
</a>

<style>
  .row {
    display: grid;
    grid-template-columns: 48px 90px minmax(0, 1fr) auto;
    grid-template-areas: 'proof state text timing';
    gap: 1rem;
    align-items: center;
  }
  .proof {
    grid-area: proof;
    width: 48px;
  }
  .state {
    grid-area: state;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.2rem;
  }
  .namecol {
    grid-area: text;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    min-width: 0;
  }
  .timing {
    grid-area: timing;
    display: flex;
    gap: 0.6rem;
    white-space: nowrap;
  }
  /* The rows are one line each wherever they are; the id is the one line
     that is never cut, so it is what tells same-workflow rows apart */
  @media (max-width: 560px) {
    .row {
      grid-template-columns: 48px minmax(0, 1fr);
      grid-template-areas:
        'proof state'
        'proof text'
        '. timing';
      gap: 0.2rem 0.7rem;
      align-items: start;
    }
    .state {
      flex-direction: row;
      align-items: center;
    }
  }
  .row.historical {
    opacity: 0.72;
  }
  .pct {
    font-family: var(--font-mono);
    font-size: var(--t-xs);
    font-variant-numeric: tabular-nums;
  }
  .line {
    font-size: var(--t-xs);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .note {
    font-style: italic;
  }
  .jobid {
    font-size: var(--t-xs);
    white-space: nowrap;
    flex: none;
  }
  /* A workflow name is what the engine resolved to run this job */
  .name {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem;
    min-width: 0;
    overflow-wrap: anywhere;
  }
  .started,
  .dur {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: var(--t-xs);
  }
  .qpos,
  .wschip {
    font-family: var(--font-mono);
    font-size: var(--t-xs);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 0 0.3rem;
  }
  .qpos {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }
  .qmove {
    display: inline-flex;
    gap: 0.1rem;
  }
  .icon {
    display: inline-flex;
    padding: 0.15rem 0.3rem;
  }
</style>
