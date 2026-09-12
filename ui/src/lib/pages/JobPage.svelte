<script lang="ts">
  import {
    Dices,
    PackageOpen,
    RotateCw,
    TriangleAlert,
    X,
  } from '@lucide/svelte'
  import { untrack } from 'svelte'
  import { ApiError, api, outputUrl, streamJobEvents } from '../api'
  import { confirmDialog } from '../confirm.svelte'
  import { go } from '../router.svelte'
  import { groupResultFiles, sectionBySubfolder } from '../results'
  import { stepProgress } from '../progress'
  import FlowView from '../editor/FlowView.svelte'
  import CopyButton from '../CopyButton.svelte'
  import DownloadLink from '../DownloadLink.svelte'
  import { notify } from '../toast'
  import type { JobDetail, JobEvent } from '../types'

  let { jobId }: { jobId: string } = $props()

  let job = $state<JobDetail | null>(null)
  // The definition this job ran, for the read-only flow view. Null while it
  // loads and when the job named a file that is no longer readable - the
  // graph is a nicety, so it simply stays absent rather than erroring.
  let definition = $state<Record<string, any> | null>(null)
  // The variable a new-seed rerun would draw into, per the server. Null for
  // a workflow that pins its seed to a literal or names none at all -
  // neither can be handed a different one, so the button stays away.
  let seedVariable = $state<string | null>(null)
  let events = $state<JobEvent[]>([])
  let error = $state('')
  // arrival clocks for pipeline_step events, for the ETA estimate
  let stepTimes = $state<number[]>([])

  const TERMINAL = ['succeeded', 'failed', 'cancelled']

  $effect(() => {
    job = null
    events = []
    definition = null
    seedVariable = null
    // Under the flat output layout two runs write the same file names, so
    // a map keyed by name would show the last job's recipe for this one
    fileMeta = {}
    // stopped guards the async gap: navigating away mid-fetch must not let
    // a late-resolving getJob open a stream nothing will ever stop
    let stopped = false
    let stop: (() => void) | null = null
    api
      .getJobWorkflow(jobId)
      .then((result) => {
        if (stopped) return
        definition = result.definition
        seedVariable = result.seed_variable
      })
      .catch(() => {
        /* no definition on file - the graph just does not appear */
      })
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

  async function rerun(newSeed: boolean) {
    try {
      go('jobs', (await api.rerunJob(jobId, newSeed)).id)
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
  // Scoped to the step now running: its phase, and its own denoise counter
  const progress = $derived(stepProgress(events as JobEvent[]))
  const denoise = $derived(progress.denoise)
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
  // Files grouped by producing step, streamed first, then confirmed by manifest
  const fileGroups = $derived(
    groupResultFiles(job?.manifest, events as JobEvent[]),
  )

  /** A prompt argument as displayable text - pipelines accept a list too. */
  function promptText(value: unknown): string {
    if (typeof value === 'string') return value
    if (Array.isArray(value))
      return value.filter((v) => typeof v === 'string').join('\n')
    return ''
  }

  // The generation metadata embedded in each image, keyed by file - per
  // file rather than per step, since each image of a batch carries its own
  // seed. Images only: the metadata route decodes an audio or video file
  // whole to probe it, and nothing it would report is shown here. A key
  // present with null is a lookup already in flight or failed
  let fileMeta = $state<Record<string, Record<string, unknown> | null>>({})
  $effect(() => {
    const workspace = job?.workspace
    const pending = fileGroups
      .flatMap((group) => group.files)
      .filter((file) => isImage(file) && !(file in untrack(() => fileMeta)))
    if (!pending.length) return
    untrack(() => {
      for (const file of pending) {
        fileMeta[file] = null
        api
          .galleryMetadata(file, workspace)
          .then((r) => {
            fileMeta[file] = r.metadata
          })
          .catch(() => {})
      }
    })
  })

  /** The gallery detail's fields for one output's embedded metadata. */
  function describe(meta: Record<string, unknown> | null | undefined) {
    const args =
      (meta?.arguments as Record<string, unknown> | undefined) ?? null
    return {
      model: typeof meta?.model_name === 'string' ? meta.model_name : '',
      seed: typeof meta?.seed === 'number' ? meta.seed : args?.seed,
      prompt: promptText(args?.prompt),
      negativePrompt: promptText(args?.negative_prompt),
    }
  }
  // Nothing at all was generated: every step the manifest lists was served
  // from the step cache. Worth saying outright - the page otherwise shows a
  // succeeded job full of images that are not this run's
  const allReused = $derived(
    fileGroups.length > 0 && fileGroups.every((group) => group.reused),
  )
  // Sections by result.subfolder - one root section, no heading, when no
  // step chose one, so an older run renders as it always did
  const sections = $derived(sectionBySubfolder(fileGroups))
  const sectioned = $derived(
    sections.length > 1 || sections[0].subfolder !== '',
  )
  const running = $derived(job !== null && !TERMINAL.includes(job.status))
  // A cancel requested while loading a model or running a task step has no
  // checkpoint to catch it until that phase finishes - without this the UI
  // goes silent for however long that takes, and looks hung rather than
  // "on its way out"
  const cancelPending = $derived(
    running && events.some((e) => e.event === 'cancel_pending'),
  )

  // Two runs of a workflow write the same file names, so the job id rides
  // along - without it the browser shows this job the image it cached from
  // the previous one. The job's own workspace rides along too, so its media
  // still loads correctly if the picker has since moved elsewhere.
  const fileUrl = (path: string) =>
    outputUrl(path, job?.id ?? '', job?.workspace)
  const isVideo = (path: string) => /\.(mp4|webm)$/i.test(path)
  const isImage = (path: string) => /\.(png|jpe?g|webp|gif)$/i.test(path)
</script>

<div class="head">
  <a href="#/jobs" class="muted">← jobs</a>
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
          {#if step === currentStep && running}
            {#if denoise}
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
            <!-- The counter tells the generating story on its own; every
                 other phase is time the bar cannot account for -->
            {#if progress.label && (!denoise || progress.phase !== 'generating')}
              <span class="muted phase" title="what this step is doing now"
                >{progress.label}</span
              >
            {/if}
          {/if}
        </div>
      {/each}
    </div>
  {/if}

  {#if definition}
    <section class="flowsection">
      <h2>Workflow</h2>
      <FlowView
        workflow={definition}
        activeStep={running ? currentStep : undefined}
        doneSteps={[...finishedSteps]}
      />
    </section>
  {/if}

  {#if fileGroups.length}
    <div class="panel">
      <h2>Results</h2>
      {#if allReused}
        <p class="muted reusednote">
          Served from the step cache: every step matched an earlier run with the
          same seed and inputs, so these are that run's files and nothing was
          generated.{#if seedVariable}
            Use <strong>New seed</strong> for a different image.{/if}
        </p>
      {/if}
      {#each sections as section (section.subfolder)}
        {#if sectioned}
          <h3 class="subhead">
            {section.subfolder === '' ? '(run root)' : `${section.subfolder}/`}
          </h3>
        {/if}
        {#each section.groups as group (group.step)}
          {#if fileGroups.length > 1}
            <svelte:element
              this={sectioned ? 'h4' : 'h3'}
              class="stephead muted"
            >
              {group.step}
              {#if group.reused && !allReused}
                <span
                  class="muted"
                  title="served from the step cache - an
                       earlier run's files, nothing generated for this step"
                  >· reused</span
                >
              {/if}
            </svelte:element>
          {/if}
          {#each group.files as file (file)}
            {@const info = describe(fileMeta[file])}
            <!-- Laid out as the gallery detail is: the media on the left,
                 what made it on the right -->
            <div class="output">
              {#if isImage(file)}
                <a
                  class="frame plain"
                  href={fileUrl(file)}
                  target="_blank"
                  title={file.split('/').pop()}
                  ><img src={fileUrl(file)} alt={file.split('/').pop()} /></a
                >
              {:else if isVideo(file)}
                <span class="frame">
                  <!-- svelte-ignore a11y_media_has_caption -->
                  <video src={fileUrl(file)} controls loop></video>
                </span>
              {:else}
                <a class="filelink" href={fileUrl(file)} target="_blank"
                  >{file.split('/').pop()}</a
                >
              {/if}
              <div class="meta">
                <div class="metabar">
                  <span class="filename">{file.split('/').pop()}</span>
                  <DownloadLink
                    href={api.outputDownloadUrl(file, job.workspace)}
                  />
                </div>
                {#if info.model}
                  <div>
                    <span class="muted">model</span>
                    <code>{info.model}</code>
                  </div>
                {/if}
                {#if info.seed !== undefined}
                  <div>
                    <span class="muted">seed</span> <code>{info.seed}</code>
                  </div>
                {/if}
                {#if info.prompt}
                  <div class="prompt">
                    <span class="muted">prompt</span>
                    <p>{info.prompt}</p>
                  </div>
                {/if}
                {#if info.negativePrompt}
                  <div class="prompt">
                    <span class="muted">negative prompt</span>
                    <p>{info.negativePrompt}</p>
                  </div>
                {/if}
              </div>
            </div>
          {/each}
        {/each}
      {/each}
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
  .reusednote {
    margin: 0 0 0.6rem;
  }

  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 1rem;
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
  .phase {
    font-size: 0.82rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .panel {
    margin-bottom: 1rem;
  }
  .flowsection {
    margin-bottom: 1rem;
  }
  .flowsection h2 {
    margin-bottom: var(--space-2);
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
  /* The step the worker is on right now - machine state, so it takes the
     signal colour rather than the interactive ink */
  .dot.active {
    background: var(--live);
    border-color: var(--live);
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
    background: var(--live);
    transition: width 0.3s;
  }
  .count {
    font-variant-numeric: tabular-nums;
    font-size: 0.8rem;
  }
  /* One output per row, the gallery detail's shape: the proof on the left,
     the recipe that made it on the right */
  .output {
    display: flex;
    gap: var(--space-4);
    align-items: flex-start;
    flex-wrap: wrap;
  }
  .output + .output {
    margin-top: var(--space-3);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line);
  }
  /* What the run produced, framed the way the catalog and the gallery
     frame it - the picture flush to its edges, no rounding of its own */
  .output :global(.frame) {
    max-width: min(480px, 100%);
  }
  .output :global(.frame > video) {
    height: auto;
  }
  .output a.filelink {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
  }
  .meta {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    font-size: 0.85rem;
    max-width: 46ch;
    min-width: 0;
  }
  .meta .muted {
    margin-right: 0.4rem;
  }
  .metabar {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.3rem;
  }
  /* The name the engine wrote, and what you would type to reference it */
  .filename {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
    overflow-wrap: anywhere;
    flex: 1;
  }
  .prompt p {
    margin: 0.15rem 0 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .error {
    color: var(--bad);
  }
  .stephead {
    font-family: var(--font-mono);
    font-weight: 600;
    line-height: 1.15;
    letter-spacing: -0.01em;
    font-size: 0.78rem;
    text-transform: none;
    margin: var(--space-3) 0 var(--space-2);
  }
  .stephead:first-of-type {
    margin-top: 0;
  }
  .subhead {
    font-size: var(--t-sm);
    text-transform: none;
    margin: var(--space-3) 0 var(--space-1);
  }
  .subhead:first-of-type {
    margin-top: 0;
  }
  .subhead + .stephead {
    margin-top: 0;
  }
</style>
