<script lang="ts">
  import { api, streamJobEvents } from '../api'
  import type { JobDetail, JobEvent } from '../types'

  let { jobId }: { jobId: string } = $props()

  let job = $state<JobDetail | null>(null)
  let events = $state<JobEvent[]>([])
  let error = $state('')

  const TERMINAL = ['succeeded', 'failed', 'cancelled']

  $effect(() => {
    job = null
    events = []
    let stop = () => {}
    api
      .getJob(jobId)
      .then((detail) => {
        job = detail
        stop = streamJobEvents(
          jobId,
          -1,
          (event) => {
            events = [...events, event]
            if (event.event === 'job_status') refresh()
          },
          () => refresh(),
        )
      })
      .catch((e) => (error = e.message))
    return () => stop()
  })

  async function refresh() {
    try {
      job = await api.getJob(jobId)
    } catch {
      /* transient */
    }
  }

  const steps = $derived(
    (events.find((e) => e.event === 'workflow_start')?.steps as string[]) ?? [],
  )
  const currentStep = $derived(
    [...events].reverse().find((e) => e.event === 'step_start')?.step as string | undefined,
  )
  const finishedSteps = $derived(
    new Set(events.filter((e) => e.event === 'step_end').map((e) => e.step as string)),
  )
  const denoise = $derived(
    [...events].reverse().find((e) => e.event === 'pipeline_step') as
      | { step: number; total_steps: number | null }
      | undefined,
  )
  const logs = $derived(
    events.filter((e) => e.event === 'log').map((e) => e.message as string),
  )
  const running = $derived(job !== null && !TERMINAL.includes(job.status))

  function fileUrl(path: string) {
    return '/outputs/' + encodeURIComponent(path.split('/').pop() ?? '')
  }
  const isVideo = (path: string) => /\.(mp4|webm)$/i.test(path)
  const isImage = (path: string) => /\.(png|jpe?g|webp|gif)$/i.test(path)
</script>

<div class="head">
  <a href="#/jobs" class="muted">← jobs</a>
  {#if job}
    <h1>{job.workflow}</h1>
    <span class="chip {job.status}">{job.status}</span>
    <span class="flex"></span>
    {#if running}
      <button class="quiet" onclick={() => api.cancelJob(jobId)}>Cancel</button>
    {/if}
  {/if}
</div>

{#if error}<p class="error">{error}</p>{/if}

{#if job}
  {#if job.warnings.length}
    <div class="panel warnings">
      {#each job.warnings as warning}<div>⚠ {warning}</div>{/each}
    </div>
  {/if}

  {#if steps.length}
    <div class="panel">
      <h2>Progress</h2>
      {#each steps as step}
        <div class="step">
          <span class="dot" class:done={finishedSteps.has(step)} class:active={step === currentStep && running}></span>
          <span class:muted={step !== currentStep && !finishedSteps.has(step)}>{step}</span>
          {#if step === currentStep && running && denoise}
            <div class="bar">
              <div
                class="fill"
                style:width={denoise.total_steps
                  ? (100 * denoise.step) / denoise.total_steps + '%'
                  : '100%'}
              ></div>
            </div>
            <span class="muted count">
              {denoise.step}{denoise.total_steps ? ` / ${denoise.total_steps}` : ''}
            </span>
          {/if}
        </div>
      {/each}
    </div>
  {/if}

  {#if job.manifest.length}
    <div class="panel">
      <h2>Results</h2>
      {#each job.manifest as entry}
        <div class="media">
          {#each entry.files as file}
            {#if isImage(file)}
              <a href={fileUrl(file)} target="_blank"><img src={fileUrl(file)} alt={entry.step} /></a>
            {:else if isVideo(file)}
              <!-- svelte-ignore a11y_media_has_caption -->
              <video src={fileUrl(file)} controls loop></video>
            {:else}
              <a href={fileUrl(file)} target="_blank">{file.split('/').pop()}</a>
            {/if}
          {/each}
        </div>
      {/each}
    </div>
  {/if}

  {#if job.error}
    <div class="panel">
      <h2 class="error">Error</h2>
      <p>{job.error}</p>
      {#if job.traceback}<pre>{job.traceback}</pre>{/if}
    </div>
  {/if}

  {#if logs.length}
    <div class="panel">
      <h2>Log</h2>
      <pre>{logs.join('\n')}</pre>
    </div>
  {/if}
{/if}

<style>
  .head { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }
  .flex { flex: 1; }
  .panel { margin-bottom: 1rem; }
  .warnings { color: var(--warn); }
  .step { display: flex; align-items: center; gap: 0.7rem; padding: 0.3rem 0; }
  .dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--panel-2); border: 1px solid var(--line);
  }
  .dot.done { background: var(--good); border-color: var(--good); }
  .dot.active { background: var(--accent); border-color: var(--accent); }
  .bar {
    flex: 1; max-width: 340px; height: 8px; border-radius: 4px;
    background: var(--panel-2); overflow: hidden;
  }
  .fill { height: 100%; background: var(--accent); transition: width 0.3s; }
  .count { font-variant-numeric: tabular-nums; font-size: 0.8rem; }
  .media { display: flex; flex-wrap: wrap; gap: 0.7rem; }
  .media img, .media video { max-width: 340px; border-radius: 6px; display: block; }
  .error { color: var(--bad); }
</style>
