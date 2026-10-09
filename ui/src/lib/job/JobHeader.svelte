<script lang="ts">
  import { Dices, PackageOpen, RotateCw, X } from '@lucide/svelte'
  import { ApiError, api } from '../api'
  import { confirmDialog } from '../confirm.svelte'
  import { go } from '../router.svelte'
  import { wsHref } from '../routes'
  import { workspace } from '../workspace.svelte'
  import CopyButton from '../CopyButton.svelte'
  import { notify } from '../toast'
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
  }: {
    job: JobDetail | null
    jobId: string
    seed: number | undefined
    runVersion: number | undefined
    running: boolean
    cancelPending: boolean
    seedVariable: string | null
  } = $props()

  async function rerun(newSeed: boolean) {
    try {
      go('ws', job!.workspace, 'jobs', (await api.rerunJob(jobId, newSeed)).id)
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

  async function exportJob() {
    if (!job || exporting) return
    exporting = true
    try {
      let result
      try {
        result = await api.exportJob(jobId, job.workspace)
      } catch (e) {
        // The server keeps one export per job. An existing one is not a
        // failure but a question, and only a 409 asks it - anything else
        // is reported as the error it is.
        if (!(e instanceof ApiError) || e.status !== 409) throw e
        const replace = await confirmDialog(
          'This job has already been exported on the server. Replace that export with a fresh one?',
          { confirmLabel: 'Replace' },
        )
        if (!replace) return
        result = await api.exportJob(jobId, job.workspace, true)
      }
      const anchor = document.createElement('a')
      anchor.href = api.exportZipUrl(result.zip_url)
      anchor.download = `${jobId}.zip`
      anchor.click()
      const mb = (result.total_bytes / (1024 * 1024)).toFixed(1)
      notify.success(
        `Exported ${result.files.length} file${result.files.length === 1 ? '' : 's'} (${mb} MB) to ${result.directory}` +
          (result.missing.length
            ? ` - ${result.missing.length} referenced file${result.missing.length === 1 ? '' : 's'} no longer on disk`
            : ''),
      )
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
    } finally {
      exporting = false
    }
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
    {#if job.device}
      <code class="muted seed" title="the card this job ran on"
        >Card: {job.device}</code
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
