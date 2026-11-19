<script lang="ts">
  import { Copy, Play, SquarePen, Trash2 } from '@lucide/svelte'
  import DownloadLink from '../DownloadLink.svelte'
  import JsonEditor from '../editor/JsonEditor.svelte'
  import VariablesForm from '../editor/VariablesForm.svelte'
  import {
    attentionBackendVariables,
    constraintHints,
    requiredVariables,
  } from '../editor'
  import { api } from '../api'
  import { goWs, route } from '../router.svelte'
  import { leafOf } from '../grouping'
  import { sharedHref, wsHref } from '../routes'
  import { workspace } from '../workspace.svelte'
  import { loadPromptLibrary } from '../promptlib.svelte'
  import { notify } from '../toast'
  import { observedText, observedTitle } from '../observed'
  import { costBasisWords, costFigure, costMeasuredOn } from '../cost'
  import { confirmDialog } from '../confirm.svelte'
  import type {
    GalleryFile,
    ValidationResult,
    WorkflowDefinition,
  } from '../types'
  import ClampedText from '../ui/ClampedText.svelte'
  import { onDestroy } from 'svelte'
  import { describePlan } from '../plan'
  import { placeFindings } from '../findings'

  let { name }: { name: string } = $props()

  let workflow = $state<WorkflowDefinition | null>(null)
  /** Where this workflow was read from, and whether it is the user's to
   * change - an examples or builtin source is read-only. Both come back on
   * `getWorkflow` itself, off the response headers, so there is no separate
   * `listWorkflows` round trip just to learn them. */
  let origin = $state('')
  let writable = $state(true)
  let overrides = $state<Record<string, string>>({})
  let error = $state('')
  let submitting = $state(false)
  /** Collapsed by default: this page is where you set variables and run,
   * and a 520px read-only editor between the form and the rest of the page
   * pushed the actual work off the screen. The editor is one click away,
   * and Edit opens the real one. */
  let showJson = $state(false)
  /** What this workflow has already made, newest first - the same proof the
   * catalog card shows, in the place where you decide whether to run it
   * again. */
  let proofs = $state<GalleryFile[]>([])
  /** This server's own measured run time for the workflow, if it has one. */
  let observed = $state<Record<string, unknown> | null>(null)
  /** The maintainer's figure for this workflow, from the same catalog card:
   * "a maintainer's figure: ~3 min · 22 GB VRAM, measured on <card> at
   * steps=30". Null when the card has no cost - never "unknown cost". */
  let curated = $state<string | null>(null)

  $effect(() => {
    overrides = {}
    workflow = null
    origin = ''
    writable = true
    loadPromptLibrary()
    api
      .getWorkflow(name)
      .then((fetched) => {
        workflow = fetched.definition
        origin = fetched.origin
        writable = fetched.writable
      })
      .catch((e) => (error = e.message))
    observed = null
    curated = null
    const wanted = name
    api
      .listWorkflows()
      .then((r) => {
        if (wanted !== name) return
        const card = r.details?.[name]
        observed = (card?.observed as Record<string, unknown>) ?? null
        const measured = card?.cost?.[0]
        if (measured) {
          const basis = costBasisWords(r.cost_basis)
          const figure = `${costFigure(measured)}, ${costMeasuredOn(card.cost, card.cost_drivers)}`
          curated = basis ? `${basis}: ${figure}` : figure
        }
      })
      .catch(() => {})
  })

  $effect(() => {
    // A run writes to <outputs>/<identity>/<run id>/, and the gallery
    // reports that identity as `folder`, so this workflow's name matches
    // its outputs directly. Loaded separately from the definition so the
    // page is usable before it answers.
    const wanted = name
    proofs = []
    api
      .gallery()
      .then((result) => {
        if (wanted !== name) return
        proofs = result.files
          .filter((file) => file.folder === wanted)
          .slice(0, 6)
      })
      .catch(() => {
        /* the page reads fine without them */
      })
  })

  const variables = $derived(Object.entries(workflow?.variables ?? {}))

  /** The arguments a Run sends: the overrides with every blank dropped, so
   * a blank field leaves the workflow's own default in place. Validate asks
   * about exactly these, so its answer predicts the Run click. */
  function runArguments(): Record<string, unknown> {
    const args: Record<string, unknown> = {}
    for (const [key, value] of Object.entries(overrides)) {
      if (value !== '') args[key] = value
    }
    return args
  }

  // The check beside Run: validate on load and after each pause in the
  // editing, with `sizes=false` so typing never asks the hub; once the form
  // has sat idle a while, one more check with sizes fills in download GB.
  // It only informs - Run stays enabled, and the server stays the gate.
  /** The quiet after the last change before checking again. */
  const CHECK_AFTER_MS = 600
  /** The idle time before the one check that asks the hub for sizes. */
  const SIZES_AFTER_MS = 3000

  let validation = $state<ValidationResult | null>(null)
  /** A check the server refused outright (an unknown name, a 400). */
  let checkError = $state('')
  let checking = $state(false)
  // Not state: bookkeeping for which answer is still wanted
  let checkTimer: ReturnType<typeof setTimeout> | undefined
  let sizesTimer: ReturnType<typeof setTimeout> | undefined
  let checkController: AbortController | null = null
  /** Bumped by every change: an answer from before the latest is dropped. */
  let generation = 0
  /** The latest call applied within this generation - a quick answer that
   * lands after the sizes one does not overwrite it. */
  let appliedCall = 0
  let callCount = 0
  let checkedFor = ''

  function stopChecks() {
    clearTimeout(checkTimer)
    clearTimeout(sizesTimer)
    checkController?.abort()
    checkController = null
  }

  function scheduleChecks(delay: number) {
    stopChecks()
    generation += 1
    appliedCall = 0
    checkController = new AbortController()
    checkTimer = setTimeout(() => check(false), delay)
    sizesTimer = setTimeout(() => check(true), SIZES_AFTER_MS)
  }

  async function check(sizes: boolean) {
    const mine = generation
    const call = ++callCount
    const signal = checkController?.signal
    checking = true
    try {
      const answer = await api.validate(
        { workflow_path: name, arguments: runArguments(), sizes },
        signal,
      )
      if (mine !== generation || call < appliedCall) return
      appliedCall = call
      validation = answer
      checkError = ''
    } catch (e) {
      if (mine !== generation || call < appliedCall) return
      if (e instanceof DOMException && e.name === 'AbortError') return
      appliedCall = call
      validation = null
      checkError = e instanceof Error ? e.message : String(e)
    } finally {
      if (mine === generation) checking = false
    }
  }

  $effect(() => {
    if (!workflow) {
      // Another workflow is loading: nothing asked about the last one is
      // wanted any more
      stopChecks()
      generation += 1
      checkedFor = ''
      checking = false
      return
    }
    // Reading every override is what makes this re-run on each change
    const asked = JSON.stringify(runArguments())
    const key = `${name}\n${asked}`
    if (key === checkedFor) return
    const first = !checkedFor.startsWith(`${name}\n`)
    checkedFor = key
    if (first) {
      validation = null
      checkError = ''
    }
    scheduleChecks(first ? 0 : CHECK_AFTER_MS)
  })

  onDestroy(stopChecks)

  const placed = $derived(
    placeFindings(
      validation,
      variables.map(([key]) => key),
    ),
  )
  const planLines = $derived(
    validation?.plan ? describePlan(validation.plan) : [],
  )
  const verdict = $derived.by(() => {
    if (checkError) return { tone: 'bad', text: 'could not check this run' }
    if (!validation) return null
    if (!validation.valid)
      return { tone: 'bad', text: 'a run would be refused' }
    if (validation.warnings.length) {
      return { tone: 'warn', text: 'a run would be admitted, with warnings' }
    }
    return { tone: 'good', text: 'a run would be admitted' }
  })

  function newFrom() {
    if (!workflow) return
    sessionStorage.setItem('dw-editor-import', JSON.stringify(workflow))
    // what the draft started from, and a file name to start with
    sessionStorage.setItem('dw-editor-import-from', name)
    sessionStorage.setItem('dw-editor-import-name', leafOf(name))
    goWs('edit')
  }

  async function remove() {
    if (
      !(await confirmDialog(
        `Delete ${name}.json? This removes the file on disk.`,
        {
          confirmLabel: 'Delete',
        },
      ))
    )
      return
    try {
      await api.deleteWorkflow(name)
      goWs('workflows')
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      notify.error(msg)
    }
  }

  async function run() {
    submitting = true
    error = ''
    try {
      const job = await api.submitJob({
        // By name, not by composed path: the server resolves a name across
        // every source it can read, so an example runs where it lives
        workflow_path: name,
        arguments: runArguments(),
      })
      goWs('jobs', job.id)
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      notify.error(msg)
    } finally {
      submitting = false
    }
  }
</script>

<a
  href={route.view.kind === 'shared'
    ? sharedHref('examples')
    : wsHref(workspace.current, 'workflows')}
  class="back muted">← workflows</a
>

<!-- One filled button: Run, the thing you came here to do. Everything else
     is outlined or bare, and Delete is pushed out of reach of Run rather
     than sitting beside it. -->
<div class="head">
  <h1>{name}</h1>
  {#if !writable}
    <span
      class="readonly muted"
      title={`read-only: this workflow comes from the ${origin} directory. Saving an edit writes a copy instead of overwriting it`}
    >
      {origin || 'read-only'}, read-only
    </span>
  {/if}
  <span class="flex"></span>
  <a
    class="plain editlink"
    href={wsHref(workspace.current, 'edit', ...name.split('/'))}
    title="open this workflow in the editor"
  >
    <SquarePen size={14} />Edit
  </a>
  <button
    class="quiet withicon"
    onclick={newFrom}
    disabled={!workflow}
    title="open a copy in the editor to save under a new name"
  >
    <Copy size={14} />Duplicate
  </button>
  <button
    class="withicon run"
    onclick={run}
    disabled={submitting || !workflow}
    title="queue this workflow with the variables below"
  >
    <Play size={14} />{submitting ? 'Submitting…' : 'Run'}
  </button>
</div>

<!-- The template's own figure, then what this server measured: two claims,
     each saying whose it is. The per-run figure for the values below is the
     plan in the run check -->
{#if curated || observedText(observed)}
  <div class="costs">
    {#if curated}
      <p class="curated num" aria-label="curated cost">
        {curated}
        <span class="muted">· other values re-price it in the plan</span>
      </p>
    {/if}
    {#if observedText(observed)}
      <p class="num muted" title={observedTitle(observed) ?? ''}>
        {observedText(observed)}
      </p>
    {/if}
  </div>
{/if}

{#if error}<p class="error">{error}</p>{/if}

{#if workflow}
  <!-- What validate says a Run with these values would do, checked as you
       edit. It informs; it never gates Run. Every finding is server text,
       shown as text -->
  <section class="panel check" aria-label="run check" aria-busy={checking}>
    <div class="verdict">
      {#if verdict}
        <span class={verdict.tone}>{verdict.text}</span>
      {/if}
      {#if checking}<span class="muted">checking…</span>{/if}
    </div>
    {#if planLines.length}
      <ul
        class="plan"
        aria-label="what a run will do"
        title="from the workflow's own cost block and this server's model cache - a plan, not a promise"
      >
        {#each planLines as line, i (i)}
          <li class:warn={line.tone === 'warn'}>{line.text}</li>
        {/each}
      </ul>
    {/if}
    {#if checkError || placed.general.length}
      <ul class="findings" aria-label="findings">
        {#if checkError}<li class="bad">{checkError}</li>{/if}
        {#each placed.general as finding, i (i)}
          <li class={finding.tone}>{finding.text}</li>
        {/each}
      </ul>
    {/if}
  </section>

  {#if workflow.description}
    <div class="muted desc prose">
      <ClampedText text={String(workflow.description)} />
    </div>
  {/if}

  <!-- What it made last time, before you decide to make another -->
  {#if proofs.length}
    <div class="proofs">
      {#each proofs as proof (proof.name)}
        <a
          class="plain frame proof"
          href={wsHref(workspace.current, 'gallery')}
          title="{proof.label} — open the gallery"
        >
          {#if proof.kind === 'image'}
            <img
              src={api.galleryThumbnailUrl(proof.name)}
              alt={proof.label}
              loading="lazy"
            />
          {:else}
            <img
              src={api.galleryThumbnailUrl(proof.name)}
              alt={proof.label}
              loading="lazy"
            />
            <span class="videobadge" aria-hidden="true"><Play size={11} /></span
            >
          {/if}
        </a>
      {/each}
    </div>
  {/if}

  {#if variables.length}
    <div class="panel">
      <!-- The heading names a thing the engine resolves, so it is mono; the
           hint is written for a person, so it is not -->
      <h2>Variables</h2>
      <p class="hint muted">
        Leave a field blank to use the workflow's own default. Type
        <code>prompt:</code>, <code>asset:</code> or <code>output:</code> to pick
        a stored prompt, an asset or an earlier run's file - or press ↓ in an empty
        field to see them.
      </p>
      <VariablesForm
        mode="override"
        variables={workflow.variables ?? {}}
        bind:overrides
        idPrefix="var-"
        backendVariables={attentionBackendVariables(workflow)}
        requiredVariables={requiredVariables(workflow)}
        constraints={constraintHints(workflow)}
        presets={workflow.presets}
        findings={placed.byVariable}
      />
    </div>
  {:else}
    <p class="muted">This workflow defines no variables.</p>
  {/if}

  <!-- The definition, the file and the destructive action all live below
       the work, in that order of how often anyone wants them -->
  <div class="foot">
    <button
      class="bare"
      onclick={() => (showJson = !showJson)}
      aria-expanded={showJson}
      title={showJson
        ? 'hide the workflow definition'
        : 'show the workflow definition'}
    >
      {showJson ? 'hide' : 'show'} JSON
    </button>
    <DownloadLink href={api.workflowDownloadUrl(name)} />
    <span class="flex"></span>
    {#if writable}
      <button
        class="quiet icon danger"
        onclick={remove}
        title="delete this workflow file from disk"
        aria-label="delete this workflow file from disk"
      >
        <Trash2 size={14} />
      </button>
    {/if}
  </div>

  {#if showJson}
    <div class="json">
      <JsonEditor
        value={JSON.stringify(workflow, null, 2)}
        readonly
        height="520px"
      />
    </div>
  {/if}
{/if}

<style>
  .back {
    display: inline-block;
    font-size: var(--t-sm);
    margin-bottom: var(--space-3);
  }
  .readonly {
    font-family: var(--font-mono);
    font-size: var(--t-xs);
    align-self: center;
  }
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.6rem;
    margin-bottom: var(--space-4);
  }
  .head h1 {
    word-break: break-word;
  }
  .head .flex {
    flex: 1;
  }
  .editlink {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: var(--t-sm);
    font-weight: 600;
    padding: 0.4rem 0.5rem;
  }
  .run {
    padding: 0.45rem 1.1rem;
  }

  /* The proofs: contact-sheet frames, sized so a row of them reads at a
     glance without taking the page over */
  .proofs {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    margin-bottom: var(--space-4);
  }
  /* .frame carries the look; this only says how big a proof is here */
  .proof {
    width: 116px;
    aspect-ratio: 1;
  }

  .foot {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
    margin-top: var(--space-5);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line);
    font-size: var(--t-sm);
  }
  .foot .flex {
    flex: 1;
  }
  .json {
    margin-top: var(--space-3);
  }
  .costs {
    margin: 0 0 var(--space-3);
  }
  .costs p {
    margin: var(--space-1) 0;
  }
  .error {
    color: var(--bad);
  }
  /* The check beside Run: the verdict, then the plan in mono (figures the
     engine resolves), then what validate said that names no field */
  .check {
    margin-bottom: var(--space-4);
    font-size: var(--t-sm);
  }
  .verdict {
    display: flex;
    gap: var(--space-3);
    align-items: baseline;
  }
  .good {
    color: var(--good);
  }
  .warn {
    color: var(--warn);
  }
  .bad {
    color: var(--bad);
  }
  .plan {
    list-style: none;
    margin: var(--space-2) 0 0;
    padding: 0;
    font-family: var(--font-mono);
    font-size: var(--t-xs);
    color: var(--muted);
    display: flex;
    flex-wrap: wrap;
    gap: 0.2rem var(--space-3);
  }
  .plan .warn {
    color: var(--warn);
  }
  .findings {
    list-style: none;
    margin: var(--space-2) 0 0;
    padding: 0;
    overflow-wrap: anywhere;
  }
  .desc {
    margin: 0 0 var(--space-4);
  }
  .hint {
    margin: -0.4rem 0 var(--space-3);
    font-size: var(--t-sm);
  }
</style>
