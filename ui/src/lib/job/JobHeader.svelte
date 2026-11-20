<script lang="ts">
  import { Dices, PackageOpen, RotateCw, X } from '@lucide/svelte'
  import { ApiError, api } from '../api'
  import { confirmDialog } from '../confirm.svelte'
  import { go } from '../router.svelte'
  import { wsHref } from '../routes'
  import { workspace } from '../workspace.svelte'
  import CopyButton from '../CopyButton.svelte'
  import RerunWithChanges from './RerunWithChanges.svelte'
  import { exportMessage } from './exportMessage'
  import { displayValue } from '../editor'
  import { formatDuration, formatExact, formatMtime, gbFromMb } from '../format'
  import { notify } from '../toast'
  import { holdKind } from '../progress'
  import type { JobDetail } from '../types'

  // The job page's title row: what ran, its state and identity, and the
  // actions a job offers - cancel while it runs, run again and export after.
  // Null while the job loads, when only the way back is shown
  let {
    job,
    jobId,
    seed,
    runVersion,
    running,
    cancelPending,
    seedVariable,
    declared = null,
  }: {
    job: JobDetail | null
    jobId: string
    seed: number | undefined
    runVersion: number | undefined
    running: boolean
    cancelPending: boolean
    seedVariable: string | null
    /** The run's definition's declared variables, for Run again with
     * changes; null while it loads or when it is not on file */
    declared?: Record<string, unknown> | null
  } = $props()

  // The request this job was run for (requests ledger). `gone` once the
  // server says 404 - the request or its whole workspace was deleted; while
  // it loads, or on any other error, the link shows
  let requestTitle = $state('')
  let requestGone = $state(false)
  // Derived primitives, so a poll that replaces `job` with an equal one
  // does not fetch the request again
  const requestWs = $derived(job?.workspace)
  const requestId = $derived(job?.request)
  $effect(() => {
    const ws = requestWs
    const id = requestId
    requestTitle = ''
    requestGone = false
    if (!ws || !id) return
    let live = true
    api.getRequest(ws, id).then(
      (found) => {
        if (live) requestTitle = found.title
      },
      (e) => {
        if (live && e instanceof ApiError && e.status === 404)
          requestGone = true
      },
    )
    return () => (live = false)
  })

  // What a rerun changed, as the server stored it: absent on a plain rerun
  const overrideKeys = $derived(Object.keys(job?.overrides ?? {}))

  // Start to finish off the job record's own clocks. A running job's
  // figure counts up on a one-second tick; the record only changes when the
  // status does, so it cannot drive the count itself
  let now = $state(Date.now() / 1000)
  $effect(() => {
    if (!running) return
    const tick = setInterval(() => (now = Date.now() / 1000), 1000)
    return () => clearInterval(tick)
  })
  const peakRam = (mb: number) =>
    mb >= 1024 ? `${gbFromMb(mb)} GB` : `${Math.round(mb)} MB`

  const took = $derived.by(() => {
    if (!job?.started_at) return null
    const end = job.finished_at ?? (running ? now : null)
    return end === null ? null : end - job.started_at
  })

  async function rerun(newSeed: boolean) {
    try {
      go(
        'ws',
        job!.workspace,
        'jobs',
        (await api.rerunJob(jobId, { newSeed })).id,
      )
    } catch (e) {
      // Without this the button silently does nothing - the failure mode
      // that made a cache-served rerun so hard to read in the first place
      notify.error(e instanceof Error ? e.message : String(e))
    }
  }

  // Export is a copy on the server before it is a download here: the
  // bundle is gathered into the workspace's exports/, then the zip the
  // server builds from it is fetched. The flag keeps a second click from
  // gathering it twice while the first is still copying media.
  let exporting = $state(false)

  /** The export, or null when one exists and the user kept it. The server
   * keeps one export per job. An existing one is not a failure but a
   * question, and only a 409 asks it - anything else is reported as the
   * error it is. */
  async function exportOrReplace(workspace: string) {
    try {
      return await api.exportJob(jobId, workspace)
    } catch (e) {
      if (!(e instanceof ApiError) || e.status !== 409) throw e
    }
    const replace = await confirmDialog(
      'This job has already been exported on the server. Replace that export with a fresh one?',
      { confirmLabel: 'Replace' },
    )
    return replace ? api.exportJob(jobId, workspace, true) : null
  }

  async function exportJob() {
    if (!job || exporting) return
    exporting = true
    try {
      const result = await exportOrReplace(job.workspace)
      if (!result) return
      const anchor = document.createElement('a')
      anchor.href = api.exportZipUrl(result.zip_url)
      anchor.download = `${jobId}.zip`
      anchor.click()
      notify.success(exportMessage(result))
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
    } finally {
      exporting = false
    }
  }

  /** Hover text for the hold, matched to whichever hold it is. */
  function holdTitle(waitingFor: string | null | undefined): string {
    const kind = holdKind(waitingFor)
    if (kind === 'host_ram')
      return "a card is free, but the host can't hold this job beside the running ones yet - it starts when one finishes"
    if (kind === 'cards_busy')
      return 'every card is running a job - this one starts when a card frees up'
    return "why this job hasn't started"
  }
</script>

<div class="head">
  <a href={wsHref(job?.workspace ?? workspace.current, 'jobs')} class="muted"
    >← jobs</a
  >
  {#if job}
    <h1>{job.workflow}</h1>
    <span class="chip {job.status}">{job.status}</span>
    <code class="muted seed" title="this job's id">{job.id}</code>
    <CopyButton text={job.id} title="copy job id" />
    {#if job?.rerun_of}
      <span class="muted"
        >rerun of <a
          class="plain"
          href={wsHref(job.workspace, 'jobs', job.rerun_of)}
          ><code>{job.rerun_of}</code></a
        ></span
      >
      {#if overrideKeys.length}
        <!-- Its own element beside "rerun of", so a sibling can be added
             without editing it -->
        <code
          class="muted seed overrides"
          title={Object.entries(job.overrides ?? {})
            .map(([name, value]) => `${name} = ${displayValue(value)}`)
            .join('\n')}>changed: {overrideKeys.join(', ')}</code
        >
      {/if}
    {/if}
    {#if job.request}
      <span class="muted">
        {#if requestGone}
          <span title="request no longer exists"
            >{job.request}{job.item ? ` › ${job.item}` : ''}</span
          >
          <span>request no longer exists</span>
        {:else}
          <a
            class="plain"
            href={wsHref(job.workspace, 'requests', job.request)}
            title={requestTitle || `request ${job.request}`}
            >{job.request}{job.item ? ` › ${job.item}` : ''}</a
          >
        {/if}
      </span>
    {/if}
    {#if cancelPending}
      <span
        class="muted"
        title="cancel requested - it takes effect once the current model load or task finishes, since neither can be interrupted mid-operation"
        >cancelling after current phase…</span
      >
    {/if}
    {#if job.queue_position !== undefined}
      <span class="muted" title="position in the waiting queue"
        >#{job.queue_position + 1} in line</span
      >
    {/if}
    {#if job.waiting_for}
      <span class="muted" title={holdTitle(job.waiting_for)}
        >{job.waiting_for}</span
      >
    {/if}
    {#if seed !== undefined}
      <code
        class="muted seed"
        title="the seed this run used - embedded in saved images alongside the recipe"
        >seed {seed}</code
      >
    {/if}
    {#if runVersion}
      <!-- The gallery labels this run's files the same way, so "version 4"
           said here finds them there -->
      <code
        class="muted seed"
        title={`version ${runVersion} of this workflow${job.run_id ? ` - run ${job.run_id}` : ''}`}
        >v{runVersion}</code
      >
    {/if}
    {#if job.started_at}
      <code
        class="muted seed"
        title={`started ${formatExact(job.started_at)}${
          job.finished_at ? `\nfinished ${formatExact(job.finished_at)}` : ''
        }`}>{formatMtime(job.started_at)}</code
      >
    {/if}
    {#if took !== null}
      <code
        class="muted seed"
        title={job.finished_at
          ? 'start to finish - excludes time queued'
          : 'running for - excludes time queued'}
        >{job.finished_at ? 'took' : 'running'} {formatDuration(took)}</code
      >
    {/if}
    {#if job.device}
      <code class="muted seed" title="the card this job ran on"
        >Card: {job.device}</code
      >
    {/if}
    {#if job.host_memory_job_peak_committed_mb != null}
      <code
        class="muted seed"
        title="the whole server process's committed memory at its peak during this job - what the host-RAM gate projects from, not growth caused by this job alone"
        >peak host RAM {peakRam(job.host_memory_job_peak_committed_mb)}</code
      >
    {/if}
    {#if job.acknowledged === 'bound'}
      <!-- Whoever queued this bound their go-ahead to a plan; the number
           they quoted is what this run was consented to at -->
      <code
        class="muted seed"
        title={`queued with a cost acknowledgement bound to the validated plan${
          job.acknowledged_cost?.fingerprint
            ? ` (${job.acknowledged_cost.fingerprint.slice(0, 15)}…)`
            : ''
        }`}
        >{job.acknowledged_cost?.minutes != null
          ? `acknowledged at ${job.acknowledged_cost.minutes} min`
          : 'acknowledged'}</code
      >
    {/if}
    <span class="flex"></span>
    {#if running}
      <button
        class="quiet withicon"
        onclick={() => api.cancelJob(jobId)}
        disabled={cancelPending}
        title={cancelPending
          ? 'cancel already requested - waiting on the current phase'
          : 'stop this run at the next step - models stay cached'}
      >
        <X size={14} />{cancelPending ? 'Cancelling…' : 'Cancel'}
      </button>
    {:else}
      <button
        class="quiet withicon"
        onclick={() => rerun(false)}
        title={seedVariable
          ? 'queue this job again with the same arguments - with the same seed, the step cache serves the files this run already made'
          : 'queue this job again with the same arguments'}
      >
        <RotateCw size={14} />Run again
      </button>
      {#if seedVariable}
        <button
          class="quiet withicon"
          onclick={() => rerun(true)}
          title="queue it again with a fresh {seedVariable} - a different image, rather than the one this run already made"
        >
          <Dices size={14} />New seed
        </button>
      {/if}
      <RerunWithChanges {job} {declared} />
      <button
        class="quiet withicon"
        onclick={exportJob}
        disabled={exporting}
        title="bundle this run - its realized workflow, manifest, the media it used and made - into the workspace's exports/ on the server, and download it as a zip"
      >
        <PackageOpen size={14} />{exporting ? 'Exporting…' : 'Export'}
      </button>
    {/if}
  {/if}
</div>

<style>
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 1rem;
    margin-bottom: 1rem;
  }
  .seed {
    font-size: 0.78rem;
  }
</style>
