<script lang="ts">
  import type { WorkflowDefinition } from '../types'
  import { Braces } from '@lucide/svelte'
  import { TERMINAL_STATUSES, api } from '../api'
  import { untrack } from 'svelte'
  import { CHANGED_MARK_MS, resource } from '../resource.svelte'
  import { followJob, jobRecord } from '../job/followJob.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import { replaceRoute, route } from '../router.svelte'
  import { wsHref } from '../routes'
  import { openInHash } from '../mediaViewer'
  import { groupResultFiles, unsavedSteps } from '../results'
  import {
    activeMember,
    chainSegments,
    finishedMembers,
    finishedNodes,
    flowNodeName,
    nodeDurations,
    stepDurations,
  } from '../runstate'
  import { estimateEta, stepProgress, activeStall } from '../progress'
  import { RunView } from '../job/runView.svelte'
  import FlowView from '../editor/FlowView.svelte'
  import JsonEditor from '../editor/JsonEditor.svelte'
  import JobHeader from '../job/JobHeader.svelte'
  import JobError from '../job/JobError.svelte'
  import JobWarnings from '../job/JobWarnings.svelte'
  import JobProgress from '../job/JobProgress.svelte'
  import NoteEditor from '../ui/NoteEditor.svelte'
  import JobResults from '../job/JobResults.svelte'
  import EventLog from '../job/EventLog.svelte'
  import ChainSegments from '../job/ChainSegments.svelte'
  import LinkedNotes from '../request/LinkedNotes.svelte'
  import { LinkedRequest } from '../request/linkedRequest.svelte'
  import RemoteNotice from '../ui/RemoteNotice.svelte'
  import type { LedgerNote } from '../types'

  let { jobId }: { jobId: string } = $props()

  // The record, kept fresh by the feed and read again when its stream
  // drops (job/followJob.svelte.ts)
  const record = jobRecord(() => jobId)
  const job = $derived(record.data ?? null)
  // The definition this job ran, for the read-only flow view: absent while
  // it loads and when the job named a file that is no longer readable - the
  // graph is a nicety, so it simply stays away rather than erroring. Its
  // `seed_variable` is what a new-seed rerun would draw into (null for a
  // seed pinned to a literal, or none), and `realized` says whether it is
  // the copy the run itself wrote, every mutable input pinned, or the
  // definition as submitted - named beside the JSON view.
  const ranRead = resource(
    async () => {
      const id = jobId
      return { ...(await api.getJobWorkflow(id)), of: id }
    },
    // With no realized copy (queued, from before run tracking, or its run
    // deleted) the server answers with the workflow file as it is now: an
    // edit to it is a workflows move. A run deleted - the realized copy
    // gone - is a gallery move. Each re-read is quiet, so another writer's
    // edit is marked; one that only swaps the submitted copy for the run's
    // realized one, or back, is not news: it moves 'realized' too
    {
      watch: ['workflows', 'gallery'],
      rows: (d) => [
        ['realized', d.realized],
        [d.of, d.definition],
      ],
    },
  )
  // A jobs move of another job reads no workflow: only this job's own
  // status moving can have changed what it ran (the run writes its
  // realized copy as its status moves). The first status seen for a job is
  // the baseline: the read for that job already answers it
  let statusSeen: { id: string; status: string } | undefined
  $effect(() => {
    const seen = job && { id: job.id, status: job.status }
    untrack(() => {
      if (
        seen &&
        statusSeen?.id === seen.id &&
        statusSeen.status !== seen.status
      )
        void ranRead.refresh()
      if (seen) statusSeen = seen
    })
  })
  const ran = $derived(ranRead.data?.of === jobId ? ranRead.data : null)
  // Another writer changed what this page shows: the inline notice
  const definitionMoved = $derived(
    ranRead.changed.has(jobId) && !ranRead.changed.has('realized'),
  )
  // The run note's box is open: its own notice says it, not the page's
  let noteEditing = $state(false)
  // Keep mine or Use theirs answered the change in the box: the line stays
  // down for the rest of the change's mark, after the box closes too
  let noteResolvedAt = $state<number | null>(null)
  function noteResolved() {
    const at = Date.now()
    noteResolvedAt = at
    setTimeout(() => {
      if (noteResolvedAt === at) noteResolvedAt = null
    }, CHANGED_MARK_MS)
  }
  const noteMoved = $derived(
    record.changed.has('note') && !noteEditing && noteResolvedAt === null,
  )
  const definition = $derived<WorkflowDefinition | null>(
    (ran?.definition as WorkflowDefinition | undefined) ?? null,
  )
  const seedVariable = $derived(ran?.seed_variable ?? null)
  const realized = $derived(ran?.realized ?? false)
  // The JSON view of that definition - off until asked
  let showJson = $state(false)
  // What the event stream has said so far (job/runView.svelte.ts)
  let run = $state.raw(new RunView())

  followJob(
    () => jobId,
    record,
    (event) => {
      run.apply(event, performance.now())
      if (
        event.event === 'job_status' ||
        event.event === 'queue_wait' ||
        (event.event === 'warning' && event.kind !== 'phase_stall')
      )
        void record.reload()
    },
    () => {
      run = new RunView()
      showJson = false
    },
  )

  // A legacy '#/jobs/<id>' lands under the last-used workspace as a first
  // guess; the job knows its own, so the URL is corrected once it answers
  $effect(() => {
    if (
      job &&
      route.view.kind === 'ws' &&
      route.view.workspace !== job.workspace
    )
      replaceRoute(wsHref(job.workspace, 'jobs', jobId))
  })

  // The request item this job is a take of: one read of the request serves
  // the header's link and the note box. Keyed on primitives, so a poll that
  // replaces `job` with an equal one does not read it again
  const linkWs = $derived(job?.workspace)
  const linkId = $derived(job?.request)
  const linkItem = $derived(job?.item ?? null)
  const linked = $derived(
    linkWs && linkId ? new LinkedRequest(linkWs, linkId) : null,
  )
  $effect(() => {
    void linked?.read()
  })
  const aboutJob = (note: LedgerNote) =>
    (note.refs ?? []).some((ref) => ref.job === jobId)
  let agentNotes = $state<ReturnType<typeof LinkedNotes>>()
  // A file names no output until its job has finished, so only a finished
  // take's files can be noted
  function noteFile(file: string) {
    if (linkItem)
      void agentNotes?.noteAbout([{ item: linkItem, job: jobId, output: file }])
  }

  const steps = $derived((run.workflowStart?.steps as string[]) ?? [])
  // The step_start the run is on, as the engine named it: a for_each member
  // keeps its '@', a sub-workflow's inner step its own name. The Progress
  // list is written in these names, so it reads off this one.
  const stepStart = $derived(run.stepStart)
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
  const finishedSteps = $derived(finishedNodes(run.marks, steps))
  // run times off the events' own clocks; a historical job streams none,
  // so its graph shows only the header's total
  const nodeTimes = $derived(nodeDurations(run.marks))
  const memberTimes = $derived(stepDurations(run.marks))
  // Scoped to the step now running: its phase, and its own denoise counter
  const progress = $derived(stepProgress(run.window))
  const denoise = $derived(progress.denoise)
  const stall = $derived(activeStall(run.last ? [run.last] : []))
  // A job read from history has no events: its record keeps the seed it was
  // queued with, under the variable the workflow draws it into
  const seed = $derived.by(() => {
    const drawn = run.workflowStart?.seed
    if (typeof drawn === 'number') return drawn
    const queued = seedVariable ? job?.arguments?.[seedVariable] : undefined
    return typeof queued === 'number' ? queued : undefined
  })
  // The record's number, or run_start's while it runs
  const runVersion = $derived(
    job?.run_version ?? (run.runStart?.version as number | undefined),
  )
  const etaSeconds = $derived(estimateEta(run.stepTimes, denoise))
  const fileGroups = $derived(groupResultFiles(job?.manifest, run.marks))

  // Steps that ran and wrote no file, with the definition's reason for each.
  // Without this a workflow that keeps most of its steps in memory shows a
  // short Results list and no word about the rest, which reads as outputs
  // that went missing rather than as a choice the workflow made
  const unsaved = $derived(unsavedSteps(job?.manifest, run.marks, definition))
  const running = $derived(
    job !== null && !TERMINAL_STATUSES.includes(job.status),
  )
  // One grain finer than the group: which entries of a for_each step have
  // finished and which is running, in the engine's own `group@entry` names
  const finishedMemberSteps = $derived(finishedMembers(run.marks))
  const segments = $derived(chainSegments(run.marks))
  const activeMemberStep = $derived(
    running ? activeMember(run.marks) : undefined,
  )
  // Where a failed run stopped: the step it had started and never ended,
  // drawn red on the graph so the reader does not have to match the error
  // to a step by name
  const failed = $derived(job?.status === 'failed')
  // A job read from history has no events, so its record's last step stands in
  const failedStep = $derived(
    activeNode ??
      flowNodeName(
        job?.progress?.step ?? undefined,
        job?.progress?.parent_step ?? undefined,
      ),
  )
  const failedNode = $derived(
    failed && failedStep && !finishedSteps.includes(failedStep)
      ? failedStep
      : undefined,
  )
  let rerunOpen = $state(false)
  // RerunWithChanges' own condition: something to change
  const canRerun = $derived(
    Object.keys({
      ...(definition?.variables ?? {}),
      ...(job?.arguments ?? {}),
    }).length > 0,
  )
  // Seen unfinished here, then failed: news to announce
  let seenRunning = $state<string | null>(null)
  $effect(() => {
    if (running) seenRunning = jobId
  })
  const failedMemberStep = $derived(
    failed ? activeMember(run.marks) : undefined,
  )

  // A graph node opens what that step left behind: its files, or its line
  // under "Steps that wrote nothing". The results carry the engine's names,
  // so a node finds its own, its for_each members' (`node@entry`) and, for a
  // sub-workflow, nothing it can name - such a node is not a control.
  function resultFor(node: string): Element | null {
    for (const el of document.querySelectorAll('[data-step]')) {
      const step = el.getAttribute('data-step') ?? ''
      if (step === node || step.startsWith(node + '@')) return el
    }
    return null
  }
  const resultSteps = $derived(
    new Set([
      ...fileGroups.map((group) => group.step),
      ...unsaved.map((entry) => entry.node),
    ]),
  )
  const hasResult = (node: string) =>
    [...resultSteps].some(
      (step) => step === node || step.startsWith(node + '@'),
    )
  function openResult(node: string) {
    resultFor(node)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }
  // A cancel requested while loading a model or running a task step has no
  // checkpoint to catch it until that phase finishes - without this the UI
  // goes silent for however long that takes, and looks hung rather than
  // "on its way out"
  const cancelPending = $derived(running && run.cancelRequested)

  // The manifest's kinds; a running job's step_end events say them first
  const kinds = $derived({ ...run.kinds, ...(job?.output_kinds ?? {}) })
</script>

<!-- A running job leads with this; a finished one has it as one line,
     after its results and the workflow -->
{#snippet progressPanel()}
  {#if steps.length}
    <div class="panel">
      <JobProgress
        {steps}
        {finishedSteps}
        {listStep}
        {running}
        {progress}
        {etaSeconds}
        {stall}
        took={job?.started_at && job.finished_at
          ? job.finished_at - job.started_at
          : null}
      />
    </div>
  {/if}
{/snippet}

<JobHeader
  {job}
  {jobId}
  {seed}
  {runVersion}
  {running}
  {cancelPending}
  {seedVariable}
  bind:rerunOpen
  declared={definition?.variables ?? null}
  request={linked?.request ?? null}
  requestGone={linked?.gone ?? false}
/>

<LoadState of={record} label="job {jobId}" what="job {jobId}">
  {#if job}
    <JobWarnings
      warnings={job.warnings}
      given={Object.keys(job.arguments ?? {})}
    />

    {#if running}{@render progressPanel()}{/if}

    {#if job.error}
      <JobError
        {job}
        step={failedNode ?? null}
        onrerun={() => (rerunOpen = true)}
        {canRerun}
        live={seenRunning === jobId}
      />
    {/if}

    {#if fileGroups.length || unsaved.length || (!running && openInHash())}
      <div class="panel">
        {#key job.id}
          <JobResults
            {job}
            {fileGroups}
            {unsaved}
            {kinds}
            {seedVariable}
            {running}
            onnote={linked?.request && linkItem && !running
              ? noteFile
              : undefined}
          />
        {/key}
      </div>
    {/if}

    <ChainSegments {job} {definition} />

    {#if job.run_dir}
      <div class="panel">
        <RemoteNotice shown={noteMoved} />
        <!-- An edit in progress keeps its draft when a newer note lands; the
             page's own save is adopted, so it marks nothing -->
        <NoteEditor
          note={job.note}
          target={job.run_dir}
          workspace={job.workspace}
          label="Your note on this job"
          bind:editing={noteEditing}
          onresolved={noteResolved}
          onsaved={(note) =>
            job && record.adopt({ ...job, note: note ?? undefined })}
        />
      </div>
    {/if}

    <!-- The note for the agent sits with the job's own, under Results:
         "Note this file" opens its box below the tiles it came from -->
    {#if linked && linkItem}
      <LinkedNotes
        bind:this={agentNotes}
        {linked}
        item={linkItem}
        filter={aboutJob}
        refs={[{ item: linkItem, job: jobId }]}
      />
    {/if}

    {#if definition}
      <section class="flowsection">
        <div class="flowhead">
          <h2>Workflow</h2>
          <span
            class="muted jsonmark"
            title={realized
              ? 'this is the realized copy the job itself wrote - every mutable input pinned to the value it used'
              : 'this is the definition as submitted - no realized copy of this job is on file'}
          >
            {realized ? 'realized' : 'as submitted'}
          </span>
          <button
            class="quiet icon"
            onclick={() => (showJson = !showJson)}
            aria-expanded={showJson}
            aria-label={showJson ? 'hide JSON' : 'show JSON'}
            title={showJson
              ? 'hide the workflow this job ran'
              : 'show the workflow this job ran, as JSON'}
          >
            <Braces size={14} />
          </button>
          <RemoteNotice shown={definitionMoved} inline />
        </div>
        <FlowView
          workflow={definition}
          activeStep={running ? activeNode : undefined}
          doneSteps={finishedSteps}
          failedStep={failedNode}
          activeMember={activeMemberStep}
          failedMember={failedMemberStep}
          onselect={openResult}
          selectable={hasResult}
          selectHint="click a step to see its results"
          doneMembers={finishedMemberSteps}
          activeSegments={segments}
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

    {#if !running}{@render progressPanel()}{/if}

    {#if run.logs.length}
      <div class="panel">
        <h2>Log</h2>
        <pre>{run.logs.join('\n')}</pre>
      </div>
    {/if}

    <EventLog {jobId} live={running} pulse={run.count} />
  {/if}
</LoadState>

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
</style>
