<script lang="ts">
  import { RotateCw, TriangleAlert, X } from '@lucide/svelte'
  import { api, streamJobEvents } from '../api'
  import { go } from '../router.svelte'
  import type { JobDetail, JobEvent } from '../types'

  let { jobId }: { jobId: string } = $props()

  let job = $state<JobDetail | null>(null)
  let events = $state<JobEvent[]>([])
  let error = $state('')
  // arrival clocks for pipeline_step events, for the ETA estimate
  let stepTimes = $state<number[]>([])

  const TERMINAL = ['succeeded', 'failed', 'cancelled']

  $effect(() => {
    job = null
    events = []
    // stopped guards the async gap: navigating away mid-fetch must not let
    // a late-resolving getJob open a stream nothing will ever stop
    let stopped = false
    let stop: (() => void) | null = null
    api
      .getJob(jobId)
      .then((detail) => {
        if (stopped) return
        job = detail
        if (detail.historical) return // no event log to stream
        stop = streamJobEvents(
          jobId,
          -1,
          (event) => {
            events.push(event)
            if (event.event === 'pipeline_step') {
              stepTimes = [...stepTimes.slice(-6), performance.now()]
            } else if (
              event.event === 'step_start' ||
              event.event === 'iteration_start'
            ) {
              // A new denoise loop: the gap since the previous loop's last
              // step includes a model load, and would inflate the ETA
              stepTimes = []
            } else if (event.event === 'job_status') {
              stepTimes = []
              refresh()
            }
          },
          () => refresh(),
        )
      })
      .catch((e) => (error = e.message))
    return () => {
      stopped = true
      stop?.()
    }
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
    [...events].reverse().find((e) => e.event === 'step_start')?.step as
      string | undefined,
  )
  const finishedSteps = $derived(
    new Set(
      events.filter((e) => e.event === 'step_end').map((e) => e.step as string),
    ),
  )
  const denoise = $derived(
    [...events].reverse().find((e) => e.event === 'pipeline_step') as
      { step: number; total_steps: number | null } | undefined,
  )
  const logs = $derived(
    events.filter((e) => e.event === 'log').map((e) => e.message as string),
  )
  const seed = $derived(
    events.find((e) => e.event === 'workflow_start')?.seed as
      number | undefined,
  )
  const etaSeconds = $derived.by(() => {
    if (!denoise?.total_steps || stepTimes.length < 3) return null
    const window = stepTimes.slice(-6)
    const perStep =
      (window[window.length - 1] - window[0]) / (window.length - 1)
    const remaining = denoise.total_steps - denoise.step
    if (remaining <= 0 || perStep <= 0) return null
    return Math.round((remaining * perStep) / 1000)
  })
  // Files stream in as steps finish - the manifest only lands at the end
  const liveFiles = $derived.by(() => {
    const manifest = job?.manifest?.flatMap((entry) => entry.files) ?? []
    const streamed = events
      .filter((e) => e.event === 'step_end')
      .flatMap((e) => (e.files as string[]) ?? [])
    return [...new Set([...manifest, ...streamed])]
  })
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
    <span class="flex"></span>
    {#if running}
      <button
        class="quiet withicon"
        onclick={() => api.cancelJob(jobId)}
        title="stop this run at the next step - models stay cached"
      >
        <X size={14} />Cancel
      </button>
    {:else}
      <button
        class="quiet withicon"
        onclick={() => api.rerunJob(jobId).then((j) => go('jobs', j.id))}
        title="queue this job again with the same arguments"
      >
        <RotateCw size={14} />Run again
      </button>
    {/if}
  {/if}
</div>

{#if error}<p class="error">{error}</p>{/if}

{#if job}
  {#if job.warnings.length}
    <div class="panel warnings warn-edge">
      {#each job.warnings as warning, i (i)}
        <div class="warnrow"><TriangleAlert size={14} /> {warning}</div>
      {/each}
    </div>
  {/if}

  {#if steps.length}
    <div class="panel">
      <h2>Progress</h2>
      {#each steps as step (step)}
        <div class="step">
          <span
            class="dot"
            class:done={finishedSteps.has(step)}
            class:active={step === currentStep && running}
          ></span>
          <span class:muted={step !== currentStep && !finishedSteps.has(step)}
            >{step}</span
          >
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
              {denoise.step}{denoise.total_steps
                ? ` / ${denoise.total_steps}`
                : ''}
              {#if etaSeconds !== null}· ~{etaSeconds}s left{/if}
            </span>
          {/if}
        </div>
      {/each}
    </div>
  {/if}

  {#if liveFiles.length}
    <div class="panel">
      <h2>Results</h2>
      <div class="media">
        {#each liveFiles as file (file)}
          {#if isImage(file)}
            <a href={fileUrl(file)} target="_blank"
              ><img src={fileUrl(file)} alt={file.split('/').pop()} /></a
            >
          {:else if isVideo(file)}
            <!-- svelte-ignore a11y_media_has_caption -->
            <video src={fileUrl(file)} controls loop></video>
          {:else}
            <a href={fileUrl(file)} target="_blank">{file.split('/').pop()}</a>
          {/if}
        {/each}
      </div>
    </div>
  {/if}

  {#if job.error}
    <div class="panel error-edge">
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
  .head {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
  }
  .flex {
    flex: 1;
  }
  .withicon {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  .seed {
    font-size: 0.78rem;
  }
  .panel {
    margin-bottom: 1rem;
  }
  .warnings {
    color: var(--warn);
  }
  .warnrow {
    display: flex;
    align-items: center;
    gap: 0.45rem;
  }
  .step {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.3rem 0;
  }
  .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--panel-2);
    border: 1px solid var(--line);
  }
  .dot.done {
    background: var(--good);
    border-color: var(--good);
  }
  .dot.active {
    background: var(--accent);
    border-color: var(--accent);
    animation: dw-pulse 1.6s ease-in-out infinite;
  }
  @media (prefers-reduced-motion: reduce) {
    .dot.active {
      animation: none;
    }
  }
  .bar {
    flex: 1;
    max-width: 340px;
    height: 8px;
    border-radius: 4px;
    background: var(--panel-2);
    overflow: hidden;
  }
  .fill {
    height: 100%;
    background: var(--accent);
    transition: width 0.3s;
  }
  .count {
    font-variant-numeric: tabular-nums;
    font-size: 0.8rem;
  }
  .media {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
  }
  .media img,
  .media video {
    max-width: 340px;
    border-radius: 6px;
    display: block;
  }
  .error {
    color: var(--bad);
  }
</style>
