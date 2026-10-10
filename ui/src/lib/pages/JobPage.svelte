<script lang="ts">
  import { TriangleAlert } from '@lucide/svelte'
  import { TERMINAL_STATUSES, api, streamJobEvents } from '../api'
  import { route } from '../router.svelte'
  import { wsHref } from '../routes'
  import { groupResultFiles, unsavedSteps } from '../results'
  import {
    activeMember,
    finishedMembers,
    finishedNodes,
    flowNodeName,
    nodeDurations,
    stepDurations,
  } from '../runstate'
  import { formatDuration } from '../format'
  import { estimateEta, nextStepTimes, stepProgress } from '../progress'
  import FlowView from '../editor/FlowView.svelte'
  import JsonEditor from '../editor/JsonEditor.svelte'
  import JobHeader from '../job/JobHeader.svelte'
  import JobProgress from '../job/JobProgress.svelte'
  import JobResults from '../job/JobResults.svelte'
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
  // Whether `definition` is the realized copy the run itself wrote (every
  // mutable input pinned) or the definition as submitted - the run predates
  // run tracking, or its run directory is gone. Named beside the JSON view.
  let realized = $state(false)
  // The JSON view of that definition - off until asked, since the flow graph
  // already answers "what did this run do" for most readers
  let showJson = $state(false)
  let events = $state<JobEvent[]>([])
  let error = $state('')
  // arrival clocks for pipeline_step events, for the ETA estimate
  let stepTimes = $state<number[]>([])

  $effect(() => {
    job = null
    events = []
    definition = null
    seedVariable = null
    realized = false
    showJson = false
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
        realized = result.realized
      })
      .catch(() => {
        /* no definition on file - the graph just does not appear */
      })
    api
      .getJob(jobId)
      .then((detail) => {
        if (stopped) return
        job = detail
        // A legacy '#/jobs/<id>' lands under the last-used workspace as a first
        // guess; the job knows its own, so the URL is corrected once it answers
        if (route.view.kind === 'ws' && route.view.workspace !== job.workspace)
          location.replace(wsHref(job.workspace, 'jobs', jobId))
        if (detail.historical) return // no event log to stream
        stop = streamJobEvents(
          jobId,
          -1,
          (event) => {
            events.push(event)
            stepTimes = nextStepTimes(stepTimes, event, performance.now())
            if (event.event === 'job_status') refresh()
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
  // The step_start the run is on, as the engine named it: a for_each member
  // keeps its '@', a sub-workflow's inner step its own name. The Progress
  // list is written in these names, so it reads off this one.
  const stepStart = $derived(
    [...events].reverse().find((e) => e.event === 'step_start'),
  )
  const currentStep = $derived(stepStart?.step as string | undefined)
  // The same step in the name the definition gives it, which is what the
  // flow graph's nodes are called - see runstate.ts for why the two differ
  const activeNode = $derived(
    flowNodeName(
      stepStart?.step as string | undefined,
      stepStart?.parent_step as string | undefined,
    ),
  )
  // A sub-workflow's inner step has no row of its own in the Progress list,
  // so what lights up while one runs is the composed step that queued it
  const listStep = $derived(
    steps.includes(currentStep ?? '') ? currentStep : activeNode,
  )
  const finishedSteps = $derived(finishedNodes(events as JobEvent[], steps))
  // run times off the events' own clocks; a historical job streams none,
  // so its graph shows only the header's total
  const nodeTimes = $derived(nodeDurations(events as JobEvent[]))
  const memberTimes = $derived(stepDurations(events as JobEvent[]))
  // Scoped to the step now running: its phase, and its own denoise counter
  const progress = $derived(stepProgress(events as JobEvent[]))
  const denoise = $derived(progress.denoise)
  // each line led by its clock - seconds since the job started, which every
  // event carries - so the log says where the time went
  const logs = $derived(
    events
      .filter((e) => e.event === 'log')
      .map((e) =>
        typeof e.at === 'number'
          ? `${('+' + formatDuration(e.at)).padStart(8)}  ${e.message}`
          : (e.message as string),
      ),
  )
  const seed = $derived(
    events.find((e) => e.event === 'workflow_start')?.seed as
      number | undefined,
  )
  // The record's number once the job has one; while it runs, the run_start
  // event says it first - so the page names the run the moment it opens
  const runVersion = $derived(
    job?.run_version ??
      (events.find((e) => e.event === 'run_start')?.version as
        number | undefined),
  )
  const etaSeconds = $derived(estimateEta(stepTimes, denoise))
  // Files grouped by producing step, streamed first, then confirmed by manifest
  const fileGroups = $derived(
    groupResultFiles(job?.manifest, events as JobEvent[]),
  )

  // Steps that ran and wrote no file, with the definition's reason for each.
  // Without this a workflow that keeps most of its steps in memory shows a
  // short Results list and no word about the rest, which reads as outputs
  // that went missing rather than as a choice the workflow made
  const unsaved = $derived(
    unsavedSteps(job?.manifest, events as JobEvent[], definition),
  )
  const running = $derived(
    job !== null && !TERMINAL_STATUSES.includes(job.status),
  )
  // One grain finer than the group: which entries of a for_each step have
  // finished and which is running, in the engine's own `group@entry` names
  const finishedMemberSteps = $derived(finishedMembers(events as JobEvent[]))
  const activeMemberStep = $derived(
    running ? activeMember(events as JobEvent[]) : undefined,
  )
  // A cancel requested while loading a model or running a task step has no
  // checkpoint to catch it until that phase finishes - without this the UI
  // goes silent for however long that takes, and looks hung rather than
  // "on its way out"
  const cancelPending = $derived(
    running && events.some((e) => e.event === 'cancel_pending'),
  )

  // The detail classifies the manifest; a running job's step_end events
  // classify each output before there is a manifest
  const kinds = $derived.by(() => {
    const found: Record<string, string | null> = {}
    for (const event of events)
      Object.assign(found, (event.output_kinds as typeof found) ?? {})
    return { ...found, ...(job?.output_kinds ?? {}) }
  })
</script>

<JobHeader
  {job}
  {jobId}
  {seed}
  {runVersion}
  {running}
  {cancelPending}
  {seedVariable}
/>

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
      <JobProgress
        {steps}
        {finishedSteps}
        {listStep}
        {running}
        {progress}
        {etaSeconds}
      />
    </div>
  {/if}

  {#if definition}
    <section class="flowsection">
      <div class="flowhead">
        <h2>Workflow</h2>
        <span
          class="muted jsonmark"
          title={realized
            ? 'this is the realized copy the run itself wrote - every mutable input pinned to the value it used'
            : 'this is the definition as submitted - no realized copy of this run is on file'}
          >{realized ? 'realized' : 'as submitted'}</span
        >
        <span class="flex"></span>
        <button
          class="bare"
          onclick={() => (showJson = !showJson)}
          aria-expanded={showJson}
          title={showJson
            ? 'hide the workflow this job ran'
            : 'show the workflow this job ran, as JSON'}
        >
          {showJson ? 'hide' : 'show'} JSON
        </button>
      </div>
      <FlowView
        workflow={definition}
        activeStep={running ? activeNode : undefined}
        doneSteps={finishedSteps}
        activeMember={activeMemberStep}
        doneMembers={finishedMemberSteps}
        {nodeTimes}
        {memberTimes}
      />
      {#if showJson}
        <div class="json">
          <JsonEditor
            value={JSON.stringify(definition, null, 2)}
            readonly
            height="520px"
          />
        </div>
      {/if}
    </section>
  {/if}

  {#if fileGroups.length || unsaved.length}
    <div class="panel">
      <h2>Results</h2>
      {#key job.id}
        <JobResults {job} {fileGroups} {unsaved} {kinds} {seedVariable} />
      {/key}
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
  .panel {
    margin-bottom: 1rem;
  }
  .flowsection {
    margin-bottom: 1rem;
  }
  .flowhead {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 1rem;
    margin-bottom: var(--space-2);
  }
  .flowhead h2 {
    margin: 0;
  }
  /* Which copy the definition is - what the engine resolves, so mono */
  .jsonmark {
    font-family: var(--font-mono);
    font-size: var(--t-xs);
  }
  .warnings {
    color: var(--warn);
  }
  .warnrow {
    display: flex;
    align-items: center;
    gap: 0.45rem;
  }
  .error {
    color: var(--bad);
  }
</style>
