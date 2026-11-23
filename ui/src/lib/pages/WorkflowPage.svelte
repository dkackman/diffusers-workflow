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
  import { SECTION_LABELS, sharedHref, wsHref } from '../routes'
  import { workspace } from '../workspace.svelte'
  import { loadPromptLibrary } from '../promptlib.svelte'
  import { notify } from '../toast'
  import { observedText, observedTitle } from '../observed'
  import { costBasisWords, costFigure, costMeasuredOn } from '../cost'
  import { confirmDialog } from '../confirm.svelte'
  import type { GalleryFile, WorkflowDefinition } from '../types'
  import ClampedText from '../ui/ClampedText.svelte'
  import { onDestroy, untrack } from 'svelte'
  import { resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import { describePlan } from '../plan'
  import { placeFindings } from '../findings'
  import { RunCheck } from '../runCheck.svelte'
  import RunCheckPanel from '../workflow/RunCheckPanel.svelte'
  import { WORKFLOW_PREFILL_KEY } from '../chainActions'
  import RemoteNotice from '../ui/RemoteNotice.svelte'

  let { name }: { name: string } = $props()

  let overrides = $state<Record<string, string>>({})
  let submitting = $state(false)
  /** Collapsed by default: this page is where you set variables and run,
   * and a 520px read-only editor between the form and the rest of the page
   * pushed the actual work off the screen. The editor is one click away,
   * and Edit opens the real one. */
  let showJson = $state(false)

  /** The definition, with where it was read from and whether it is the
   * user's to change - an examples or builtin source is read-only. Both
   * come back on `getWorkflow` itself, off the response headers, so there
   * is no separate `listWorkflows` round trip just to learn them. Held only
   * while it is this name's. Keyed by name, so a write from elsewhere marks
   * it changed and the page says so */
  const read = resource(
    async () => {
      const wanted = name
      return { name: wanted, ...(await api.getWorkflow(wanted)) }
    },
    { watch: ['workflows'], rows: (d) => [[d.name, d.definition]] },
  )
  const fetched = $derived(read.data?.name === name ? read.data : null)
  const workflow = $derived<WorkflowDefinition | null>(
    fetched?.definition ?? null,
  )
  const origin = $derived(fetched?.origin ?? '')
  const writable = $derived(fetched?.writable ?? true)
  /** What this workflow has already made, newest first - the same proof the
   * catalog card shows, in the place where you decide whether to run it
   * again. A run writes to <outputs>/<identity>/<run id>/, and the gallery
   * reports that identity as `folder`, so this workflow's name matches its
   * outputs directly. */
  const proofRead = resource(
    async () => {
      const wanted = name
      const result = await api.gallery()
      const files = result.files.filter(
        (file: GalleryFile) => file.folder === wanted,
      )
      return { name: wanted, files: files.slice(0, 6) }
    },
    { watch: ['gallery'] },
  )
  const proofs = $derived(
    proofRead.data?.name === name ? proofRead.data.files : [],
  )
  /** The catalog card: this server's own measured run time for the
   * workflow, and the maintainer's figure - "a maintainer's figure: ~3 min
   * · 22 GB VRAM, measured on <card> at steps=30". Null when the card has
   * no cost - never "unknown cost". */
  const catalog = resource(() => api.listWorkflows(), { watch: ['workflows'] })
  const card = $derived(catalog.data?.details?.[name])
  const observed = $derived(
    (card?.observed as Record<string, unknown> | undefined) ?? null,
  )
  const curated = $derived.by(() => {
    const measured = card?.cost?.[0]
    if (!measured || !card) return null
    const basis = costBasisWords(catalog.data?.cost_basis)
    const figure = `${costFigure(measured)}, ${costMeasuredOn(card.cost, card.cost_drivers)}`
    return basis ? `${basis}: ${figure}` : figure
  })

  /** Values another page asked this one to open with (Repair this
   * segment, #1476): read once, and only when they name this workflow.
   * Filled in, never submitted - Run is still the user's */
  function takePrefill(): Record<string, string> {
    const raw = sessionStorage.getItem(WORKFLOW_PREFILL_KEY)
    if (raw === null) return {}
    sessionStorage.removeItem(WORKFLOW_PREFILL_KEY)
    try {
      const parsed = JSON.parse(raw)
      if (parsed?.workflow !== name) return {}
      const args = parsed.arguments
      if (!args || typeof args !== 'object') return {}
      return Object.fromEntries(
        Object.entries(args).map(([key, value]) => [key, String(value)]),
      )
    } catch {
      return {}
    }
  }

  $effect(() => {
    void name
    untrack(() => {
      overrides = takePrefill()
      showJson = false
      loadPromptLibrary()
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

  /** The check beside Run, asked about exactly what Run would send. */
  const runCheck = new RunCheck((sizes, signal) =>
    api.validate(
      { workflow_path: name, arguments: runArguments(), sizes },
      signal,
    ),
  )

  $effect(() => {
    if (!workflow) {
      runCheck.clear()
      return
    }
    // Reading every override is what makes this re-run on each change
    runCheck.changed(name, runArguments())
  })

  onDestroy(runCheck.stop)

  const placed = $derived(
    placeFindings(
      runCheck.validation,
      variables.map(([key]) => key),
    ),
  )
  const planLines = $derived(
    runCheck.validation?.plan ? describePlan(runCheck.validation.plan) : [],
  )

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
      notify.failure(`delete ${name}`, e)
    }
  }

  async function run() {
    submitting = true
    try {
      const job = await api.submitJob({
        // By name, not by composed path: the server resolves a name across
        // every source it can read, so an example runs where it lives
        workflow_path: name,
        arguments: runArguments(),
      })
      goWs('jobs', job.id)
    } catch (e) {
      notify.failure(`run ${name}`, e)
    } finally {
      submitting = false
    }
  }
</script>

<!-- One filled button: Run, the thing you came here to do. Everything else
     is outlined or bare, and Delete is pushed out of reach of Run rather
     than sitting beside it. -->
<PageHeader
  title={name}
  back={route.view.kind === 'shared'
    ? { href: sharedHref('examples'), label: SECTION_LABELS.shared.examples }
    : {
        href: wsHref(workspace.current, 'workflows'),
        label: SECTION_LABELS.ws.workflows,
      }}
>
  {#if !writable}
    <span
      class="readonly muted"
      title={`read-only: this workflow comes from the ${origin} directory. Saving an edit writes a copy instead of overwriting it`}
    >
      {origin || 'read-only'}, read-only
    </span>
  {/if}
  {#snippet actions()}<a
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
    </button>{/snippet}
</PageHeader>

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

<LoadState of={read} label={name} what={name}>
  {#if workflow}
    <RemoteNotice shown={read.changed.has(name)} />
    <RunCheckPanel
      verdict={runCheck.verdict}
      checking={runCheck.checking}
      {planLines}
      checkError={runCheck.error}
      findings={placed.general}
    />

    {#if workflow.description}
      <div class="muted desc prose">
        <ClampedText text={String(workflow.description)} />
      </div>
    {/if}

    <!-- What it made last time, before you decide to make another -->
    <LoadState of={proofRead} label="what {name} made" quiet>
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
                <span class="videobadge" aria-hidden="true">
                  <Play size={11} />
                </span>
              {/if}
            </a>
          {/each}
        </div>
      {/if}
    </LoadState>

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
</LoadState>

<style>
  .readonly {
    font-family: var(--font-mono);
    font-size: var(--t-xs);
    align-self: center;
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
  .desc {
    margin: 0 0 var(--space-4);
  }
  .hint {
    margin: -0.4rem 0 var(--space-3);
    font-size: var(--t-sm);
  }
</style>
