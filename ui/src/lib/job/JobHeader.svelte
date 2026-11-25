<script lang="ts">
  import { Dices, PackageOpen, RotateCw, X } from '@lucide/svelte'
  import StatusChip from '../ui/StatusChip.svelte'
  import { ApiError, api } from '../api'
  import { action } from '../action.svelte'
  import { confirmDialog } from '../confirm.svelte'
  import ActionButton from '../ui/ActionButton.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import { go } from '../router.svelte'
  import { SECTION_LABELS, wsHref } from '../routes'
  import { workspace } from '../workspace.svelte'
  import CopyButton from '../CopyButton.svelte'
  import RerunWithChanges from './RerunWithChanges.svelte'
  import { exportMessage } from './exportMessage'
  import { displayValue } from '../editor'
  import { formatDuration, formatExact, gbFromMb } from '../format'
  import Time from '../ui/Time.svelte'
  import { notify } from '../toast'
  import { holdKind, holdText } from '../progress'
  import { versionTitle, workflowName } from '../terms'
  import type { JobDetail, LedgerRequest } from '../types'

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
    rerunOpen = $bindable(false),
    declared = null,
    request = null,
    requestGone = false,
  }: {
    job: JobDetail | null
    jobId: string
    seed: number | undefined
    runVersion: number | undefined
    running: boolean
    cancelPending: boolean
    seedVariable: string | null
    /** Whether Run again with changes is open - the error panel opens it */
    rerunOpen?: boolean
    /** The run's definition's declared variables, for Run again with
     * changes; null while it loads or when it is not on file */
    declared?: Record<string, unknown> | null
    /** The request the job was run for, once read */
    request?: LedgerRequest | null
    /** The server said that request is gone (404) */
    requestGone?: boolean
  } = $props()

  // The request this job was run for (requests ledger) names the link.
  // The page reads it, for this title and its note box alike
  const requestTitle = $derived(request?.title ?? '')

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

  // The file the job ran names it; the JSON id beside it only when it says
  // something else
  const leaf = $derived(job ? workflowName(job) : jobId)
  const fileName = $derived(job ? workflowName(job) : 'Job')
  const jsonId = $derived(
    job?.workflow && job.workflow !== fileName ? job.workflow : undefined,
  )

  // A rerun failing must say so - the silent button was what made a
  // cache-served rerun so hard to read in the first place
  const rerun = action(
    async (newSeed: boolean) => {
      const next = await api.rerunJob(jobId, { newSeed })
      go('ws', job!.workspace, 'jobs', next.id)
    },
    { doing: 'run it again' },
  )

  // A cancel loses the work so far, which only matters once there is some
  const CANCEL_ASKS_AFTER_S = 60
  const cancel = action(() => api.cancelJob(jobId), {
    doing: () => `cancel ${leaf}`,
    confirm: () =>
      job?.started_at && now - job.started_at > CANCEL_ASKS_AFTER_S
        ? {
            message: `Cancel ${leaf}? It has run for ${formatDuration(now - job.started_at)}; the work so far is lost.`,
            confirmLabel: 'Cancel job',
            cancelLabel: 'Keep running',
            danger: true,
          }
        : null,
  })

  // Export is a copy on the server before it is a download here: the
  // bundle is gathered into the workspace's exports/, then the zip the
  // server builds from it is fetched. The action's busy guard keeps a second
  // click from gathering it twice while the first is still copying media.

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
      { title: 'Replace the export?', confirmLabel: 'Replace', danger: true },
    )
    return replace ? api.exportJob(jobId, workspace, true) : null
  }

  const exportJob = action(
    async () => {
      const result = await exportOrReplace(job!.workspace)
      if (!result) return
      const anchor = document.createElement('a')
      anchor.href = api.exportZipUrl(result.zip_url)
      anchor.download = `${jobId}.zip`
      anchor.click()
      notify.success(exportMessage(result))
    },
    { doing: 'export this job' },
  )

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

<PageHeader
  title={fileName}
  secondary={jsonId}
  back={{
    href: wsHref(job?.workspace ?? workspace.current, 'jobs'),
    label: SECTION_LABELS.ws.jobs,
  }}
>
  {#if job}
    <StatusChip status={job.status} />
    <code class="muted seed" title="this job's id">{job.id}</code>
    <CopyButton text={job.id} title="copy job id" />
    {#if job?.rerun_of}
      <span class="muted">
        rerun of <a
          class="plain"
          href={wsHref(job.workspace, 'jobs', job.rerun_of)}
        >
          <code>{job.rerun_of}</code>
        </a>
      </span>
      {#if overrideKeys.length}
        <!-- Its own element beside "rerun of", so a sibling can be added
             without editing it -->
        <code
          class="muted seed overrides"
          title={Object.entries(job.overrides ?? {})
            .map(([name, value]) => `${name} = ${displayValue(value)}`)
            .join('\n')}
        >
          changed: {overrideKeys.join(', ')}
        </code>
      {/if}
    {/if}
    {#if job.request}
      <span class="muted">
        {#if requestGone}
          <span title="request no longer exists">
            {job.request}{job.item ? ` › ${job.item}` : ''}
          </span>
          <span>request no longer exists</span>
        {:else}
          <a
            class="plain"
            href={wsHref(job.workspace, 'requests', job.request)}
            title={requestTitle || `request ${job.request}`}
          >
            {job.request}{job.item ? ` › ${job.item}` : ''}
          </a>
        {/if}
      </span>
    {/if}
    {#if cancelPending}
      <span
        class="muted"
        title="cancel requested - it takes effect once the current model load or task finishes, since neither can be interrupted mid-operation"
      >
        cancelling after current phase…
      </span>
    {/if}
    {#if job.queue_position !== undefined}
      <span class="muted" title="position in the waiting queue">
        #{job.queue_position + 1} in line
      </span>
    {/if}
    {#if job.waiting_for}
      <span class="muted" title={holdTitle(job.waiting_for)}>
        {holdText(job.waiting_for)}
      </span>
    {/if}
    {#if seed !== undefined}
      <code
        class="muted seed"
        title="the seed this job used - embedded in saved images alongside the recipe"
      >
        seed {seed}
      </code>
    {/if}
    {#if runVersion}
      <!-- The gallery labels this job's files the same way, so "v4" said
           here finds them there -->
      <code class="muted seed" title={versionTitle(runVersion)}>
        v{runVersion}
      </code>
    {/if}
    {#if job.started_at}
      <code class="muted seed">
        <Time
          at={job.started_at}
          mode="exact"
          title={`started ${formatExact(job.started_at)}${
            job.finished_at ? `\nfinished ${formatExact(job.finished_at)}` : ''
          }`}
        />
      </code>
    {/if}
    {#if took !== null}
      <code
        class="muted seed"
        title={job.finished_at
          ? 'start to finish - excludes time queued'
          : 'running for - excludes time queued'}
      >
        {job.finished_at ? 'took' : 'running'}
        {formatDuration(took)}
      </code>
    {/if}
    {#if job.device}
      <code class="muted seed" title="the card this job ran on">
        Card: {job.device}
      </code>
    {/if}
    {#if job.host_memory_job_peak_committed_mb != null}
      <code
        class="muted seed"
        title="the whole server process's committed memory at its peak during this job - what the host-RAM gate projects from, not growth caused by this job alone"
      >
        peak host RAM {peakRam(job.host_memory_job_peak_committed_mb)}
      </code>
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
      >
        {job.acknowledged_cost?.minutes != null
          ? `acknowledged at ${job.acknowledged_cost.minutes} min`
          : 'acknowledged'}
      </code>
    {/if}
  {/if}
  {#snippet actions()}{#if job}
      {#if running}
        <ActionButton
          act={cancel}
          disabled={cancelPending}
          title={cancelPending
            ? 'cancel already requested - waiting on the current phase'
            : 'stop this job at the next step - models stay cached'}
        >
          <X size={14} />{cancelPending ? 'Cancelling…' : 'Cancel'}
        </ActionButton>
      {:else}
        <ActionButton
          act={rerun}
          args={[false]}
          title={seedVariable
            ? 'queue this job again with the same arguments - with the same seed, the step cache serves the files this job already made'
            : 'queue this job again with the same arguments'}
        >
          <RotateCw size={14} />Run again
        </ActionButton>
        {#if seedVariable}
          <ActionButton
            act={rerun}
            args={[true]}
            title="queue it again with a fresh {seedVariable} - a different image, rather than the one this job already made"
          >
            <Dices size={14} />New seed
          </ActionButton>
        {/if}
        <RerunWithChanges
          {job}
          {declared}
          {seedVariable}
          bind:open={rerunOpen}
        />
        <ActionButton
          act={exportJob}
          title="bundle this job - its realized workflow, manifest, the media it used and made - into the workspace's exports/ on the server, and download it as a zip"
        >
          <PackageOpen size={14} />{exportJob.busy ? 'Exporting…' : 'Export'}
        </ActionButton>
      {/if}
    {/if}{/snippet}
</PageHeader>

<style>
  .seed {
    font-size: 0.78rem;
  }
</style>
