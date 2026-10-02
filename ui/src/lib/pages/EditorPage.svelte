<script lang="ts">
  import { editorLists, loadEditorLists } from '../editorLists.svelte'
  import { DocumentEditor, NEW_FOLDER } from '../editorShell.svelte'
  import ViewSwitch from '../editor/ViewSwitch.svelte'
  import FolderPicker from '../editor/FolderPicker.svelte'
  import EditorBody from '../editor/EditorBody.svelte'
  import {
    ChevronUp,
    CircleCheck,
    Columns2,
    Braces,
    FileCog,
    LayoutList,
    Play,
    Plus,
    Save,
    TriangleAlert,
    Workflow,
    X,
  } from '@lucide/svelte'
  import { api } from '../api'
  import { writableRoot } from '../libraries'
  import { notify } from '../toast'
  import { confirmDialog } from '../confirm.svelte'
  import { goWs } from '../router.svelte'
  import { wsHref } from '../routes'
  import { workspace } from '../workspace.svelte'
  import {
    emptyWorkflow,
    emptyStep,
    emptyTaskStep,
    emptyWorkflowStep,
    referenceSuggestions,
  } from '../editor'
  import { danglingReferenceDetails, flowGraph } from '../flow'
  import { groupOf, leafOf } from '../grouping'
  import { loadPromptLibrary, promptLibrary } from '../promptlib.svelte'
  import { storageGet, storageSet } from '../storage'
  import { describePlan } from '../plan'
  import StepEditor from '../editor/StepEditor.svelte'
  import VariablesForm from '../editor/VariablesForm.svelte'
  import FlowView from '../editor/FlowView.svelte'
  import type { ValidationResult, WorkflowDefinition } from '../types'

  let { name = '' }: { name?: string } = $props()

  const ed = new DocumentEditor({
    viewKey: 'dw-editor-view',
    views: ['form', 'split', 'json', 'flow'],
    // migrate the old boolean split preference
    legacyView: () =>
      localStorage.getItem('dw-editor-split') === '1' ? 'split' : null,
  })
  ed.doc = emptyWorkflow()
  let workflowDir = $state('')
  // The file fields collapse behind the path chip once the workflow has a
  // name - they are settings, not something you retouch on every edit
  let fileOpen = $state(false)

  type StepMode = 'collapsed' | 'compact' | 'full'
  // Keyed by step name; existing steps open compact, new ones full
  let stepModes = $state<Record<string, StepMode>>({})
  const modesKey = $derived(`step-modes:${name || '(unsaved)'}`)

  function modeOf(step: Record<string, any>): StepMode {
    return stepModes[step.name] ?? 'compact'
  }
  function setMode(step: Record<string, any>, mode: StepMode) {
    stepModes[step.name] = mode
    storageSet(modesKey, $state.snapshot(stepModes))
  }
  function setAllModes(mode: StepMode) {
    for (const step of ed.doc.steps ?? []) stepModes[step.name] = mode
    storageSet(modesKey, $state.snapshot(stepModes))
  }

  // Existing folders, from the listing - one level is the designed depth,
  // but any deeper directories that exist still appear and keep working
  const folders = $derived(
    [...new Set(editorLists.workflowFiles.map(groupOf).filter(Boolean))].sort(),
  )
  let validation = $state<ValidationResult | null>(null)
  const VIEWS = [
    {
      view: 'form',
      label: 'form',
      title: 'edit with introspection-driven forms',
      icon: LayoutList,
    },
    {
      view: 'split',
      label: 'split',
      title: 'form beside the JSON - both editable, blur applies',
      icon: Columns2,
    },
    {
      view: 'json',
      label: 'JSON',
      title: 'edit the raw JSON, schema-aware',
      icon: Braces,
    },
    {
      view: 'flow',
      label: 'flow',
      title:
        'read-only data-flow diagram: steps as boxes, previous_result as edges',
      icon: Workflow,
    },
  ] as const

  // Crumbs back out of the editor. The read-only page is where an edit
  // usually starts, and until now the only exit landed on the list - so
  // the last crumb links back to the workflow itself, when one is saved.
  const crumbFolders = $derived(
    name ? groupOf(name).split('/').filter(Boolean) : [],
  )
  const crumbName = $derived(name ? leafOf(name) : '')
  const workflowHref = $derived(
    wsHref(workspace.current, 'workflows', ...name.split('/')),
  )

  // Leaving with unsaved edits used to drop them without a word. The
  // confirm is async, so the default navigation is always prevented first
  // and replayed by hand once the answer comes back.
  async function confirmLeave(event: MouseEvent) {
    if (!ed.dirty) return
    event.preventDefault()
    const target = (event.currentTarget as HTMLAnchorElement).href
    if (await confirmDialog('Discard unsaved changes?')) {
      window.location.href = target
    }
  }

  const savePreview = $derived.by(() => {
    const directory = ed.directory()
    const file = ed.saveName || 'unnamed'
    return `${workflowDir}/${directory ? directory + '/' : ''}${file}.json`
  })
  // The data-flow graph and per-step reference problems drive the rail's
  // producer/consumer chips and inline warnings - promptLibrary.names stays
  // undefined until the listing lands, so a missing library must not flag
  // every prompt: reference as dangling
  const flow = $derived(flowGraph($state.snapshot(ed.doc)))
  const problemsByStep = $derived.by(() => {
    const details = danglingReferenceDetails(
      $state.snapshot(ed.doc),
      promptLibrary.names,
    )
    const grouped: Record<number, string[]> = {}
    for (const d of details) {
      grouped[d.stepIndex] = [...(grouped[d.stepIndex] ?? []), d.message]
    }
    return new Map(
      Object.entries(grouped).map(([index, messages]) => [
        Number(index),
        messages,
      ]),
    )
  })
  let hovered = $state<string | null>(null)
  // Memoized so datalist options keep stable DOM identity - churn on every
  // render made the browser's suggestion dropdown flaky on first focus
  const stepReferences = $derived.by(() => {
    const snapshot = $state.snapshot(ed.doc)
    return ((snapshot.steps as unknown[]) ?? []).map((_, index) =>
      referenceSuggestions(snapshot, index, promptLibrary.names ?? []),
    )
  })

  $effect(() => {
    loadEditorLists()
    api.listWorkflows().then((r) => {
      editorLists.workflowFiles = r.workflows.map((file) => `${file}.json`)
      workflowDir = writableRoot(r.libraries)
    })
    loadPromptLibrary()
    validation = null
    if (name) {
      ed.saveName = leafOf(name)
      ed.folder = groupOf(name)
      fileOpen = false
      api
        .getWorkflow(name)
        .then((fetched) => {
          ed.load(fetched.definition as WorkflowDefinition)
          stepModes = storageGet(modesKey, {})
        })
        .catch((e) => notify.error(e.message))
    } else {
      // A gallery "open as workflow" hands the definition over in
      // sessionStorage - one-shot, so a plain "New" stays a blank slate
      const imported = ed.takeImport('dw-editor-import', 'dw-editor-folder')
      // Built as a plain local object and assigned once: reading the
      // workflow proxy here would subscribe this effect to every edit and
      // re-fire all the listing calls on each keystroke
      const fresh = imported ?? (emptyWorkflow() as Record<string, any>)
      if (imported) notify.success('Imported from image metadata')
      ed.load(fresh)
      ed.saveName = ''
      // A new workflow has nowhere to save to yet - show the fields
      fileOpen = true
      stepModes = imported
        ? storageGet(modesKey, {})
        : { [fresh.steps[0].name]: 'full' }
    }
  })

  function onKeydown(event: KeyboardEvent) {
    if (!(event.ctrlKey || event.metaKey)) return
    if (event.key === 's') {
      event.preventDefault()
      save()
    } else if (event.key === 'Enter') {
      event.preventDefault()
      run()
    }
  }

  function addStep(kind: string) {
    const step =
      kind === 'task'
        ? emptyTaskStep()
        : kind === 'workflow'
          ? emptyWorkflowStep()
          : emptyStep()
    // Step mode and flow-graph edges are keyed by name - two steps sharing
    // the factory default (e.g. 'generate') would collide in stepModes
    const existingNames = new Set(
      (ed.doc.steps ?? []).map((s: Record<string, any>) => s.name),
    )
    if (existingNames.has(step.name)) {
      let n = 2
      while (existingNames.has(`${step.name}-${n}`)) n++
      step.name = `${step.name}-${n}`
    }
    ed.doc.steps = [...(ed.doc.steps ?? []), step]
    stepModes[step.name] = 'full'
  }

  function removeStep(index: number) {
    ed.doc.steps.splice(index, 1)
  }

  function moveStep(index: number, delta: number) {
    const steps = ed.doc.steps
    const target = index + delta
    if (target < 0 || target >= steps.length) return
    ;[steps[index], steps[target]] = [steps[target], steps[index]]
  }

  async function validate(): Promise<boolean> {
    ed.busy = true
    try {
      validation = await api.validate(
        $state.snapshot(ed.doc) as WorkflowDefinition,
      )
      return validation.valid
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
      return false
    } finally {
      ed.busy = false
    }
  }

  async function save() {
    const path = ed.savePath()
    if (!path) {
      if (!ed.saveName) notify.error('Give the workflow a file name first')
      else if (!ed.directory()) notify.error('Name the new folder first')
      else notify.error('Folder names: letters, numbers, dot, dash, underscore')
      // The message names a field the user cannot see while collapsed
      fileOpen = true
      return
    }
    if (!(await validate())) return
    ed.busy = true
    try {
      await api.saveWorkflow(
        path,
        $state.snapshot(ed.doc) as WorkflowDefinition,
      )
      ed.commitNewFolder()
      // The folder picker's options come from the listing - a folder this
      // save just created must appear there, or the select falls back to
      // "(root)" while the state still names the folder
      if (!editorLists.workflowFiles.includes(`${path}.json`)) {
        editorLists.workflowFiles = [
          ...editorLists.workflowFiles,
          `${path}.json`,
        ]
      }
      ed.markSaved()
      notify.success(`Saved to ${path}`)
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
    } finally {
      ed.busy = false
    }
  }

  async function run() {
    if (!(await validate())) return
    ed.busy = true
    try {
      // base_dir anchors relative paths (images, sub-workflow files) the
      // way running the saved file would - at the workflow's own folder
      const directory =
        ed.folder && ed.folder !== NEW_FOLDER ? `/${ed.folder}` : ''
      const job = await api.submitJob({
        workflow: $state.snapshot(ed.doc) as WorkflowDefinition,
        base_dir: `${workflowDir}${directory}`,
      })
      goWs('jobs', job.id)
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
    } finally {
      ed.busy = false
    }
  }

  // The flow view's click-to-navigate: land back on the form view with the
  // step's row scrolled into place and briefly lit, the same highlight the
  // rail chips already use on hover - a reasonable cross-reference without
  // teaching the flow view anything about editing.
  function focusStep(stepName: string) {
    ed.setView('form')
    hovered = stepName
    requestAnimationFrame(() => {
      document
        .getElementById('step-' + stepName)
        ?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    })
    setTimeout(() => {
      if (hovered === stepName) hovered = null
    }, 2000)
  }
</script>

<svelte:window onkeydown={onKeydown} />

<div class="head">
  <nav class="crumbs muted" aria-label="breadcrumb">
    <a href={wsHref(workspace.current, 'workflows')} onclick={confirmLeave}
      >← workflows</a
    >
    {#each crumbFolders as part (part)}
      <span class="sep">/</span><span>{part}</span>
    {/each}
    {#if crumbName}
      <span class="sep">/</span>
      <a
        href={workflowHref}
        onclick={confirmLeave}
        title="back to the read-only view of this workflow"
      >
        {crumbName}
      </a>
    {/if}
  </nav>
  <label class="wfidwrap">
    <span class="wfidlabel">id</span>
    <input
      class="wfid"
      bind:value={ed.doc.id}
      placeholder="workflow id"
      aria-label="workflow id"
      title="the workflow's id - how it names itself, independent of the file name"
    />
  </label>
  {#if ed.dirty}
    <span
      class="chip unsaved"
      title="this definition differs from the last save">unsaved</span
    >
  {/if}
  <span class="flex"></span>
  <ViewSwitch
    view={ed.view}
    options={[...VIEWS]}
    onselect={(view) => ed.setView(view)}
  />
  <button
    class="quiet withicon"
    onclick={validate}
    disabled={ed.busy}
    title="check against the schema and real pipeline signatures, without running"
  >
    <CircleCheck size={14} />Validate
  </button>
  <button
    class="quiet withicon"
    class:dirtybtn={ed.dirty}
    onclick={save}
    disabled={ed.busy}
    title="validate, then write to the workflow directory under the name below (Ctrl+S)"
  >
    <Save size={14} />Save
  </button>
  <button
    class="withicon"
    onclick={run}
    disabled={ed.busy}
    title="validate, then queue this definition as a job - no save needed (Ctrl+Enter)"
  >
    <Play size={14} />Run
  </button>
</div>

{#if fileOpen}
  <div class="filebar panel">
    <div class="filegrid">
      <label for="wf-folder">folder</label>
      <div class="folderrow">
        <FolderPicker
          id="wf-folder"
          bind:folder={ed.folder}
          bind:newFolder={ed.newFolder}
          {folders}
          newFolderTitle="name for the new folder at the root of the workflow directory"
        />
      </div>

      <label for="wf-savename">file name</label>
      <div class="namerow">
        <input
          id="wf-savename"
          class="savename"
          bind:value={ed.saveName}
          placeholder="MyWorkflow"
        />
        <span class="muted">.json</span>
      </div>

      <label for="wf-description">description</label>
      <input
        id="wf-description"
        spellcheck="true"
        value={ed.doc.description ?? ''}
        placeholder="shown on the workflow card"
        title="a short description of what this workflow does"
        onchange={(e) => {
          const v = e.currentTarget.value
          if (v) ed.doc.description = v
          else delete ed.doc.description
        }}
      />
    </div>
    <div class="filefoot">
      <span class="muted path">{savePreview}</span>
      <button
        class="quiet withicon"
        onclick={() => (fileOpen = false)}
        disabled={!ed.saveName}
        title="collapse the file settings"
      >
        <ChevronUp size={14} />done
      </button>
    </div>
  </div>
{:else}
  <div class="savebar">
    <button
      class="quiet withicon pathchip"
      onclick={() => (fileOpen = true)}
      title="change the folder, file name or description"
    >
      <FileCog size={14} /><span class="path">{savePreview}</span>
    </button>
    {#if ed.doc.description}
      <span class="muted desc">{ed.doc.description}</span>
    {/if}
  </div>
{/if}

{#if validation}
  <div
    class="panel validation"
    class:error-edge={!validation.valid}
    class:warn-edge={validation.valid && validation.warnings.length > 0}
    class:good-edge={validation.valid && validation.warnings.length === 0}
  >
    <button
      class="quiet icon dismiss"
      onclick={() => (validation = null)}
      title="dismiss"
      aria-label="dismiss validation results"
    >
      <X size={13} />
    </button>
    {#if validation.valid && !validation.warnings.length}
      <span class="ok"
        ><CircleCheck size={14} /> schema-valid, no argument warnings</span
      >
    {:else if validation.valid}
      {#each validation.warnings as warning, i (i)}
        <div class="warn"><TriangleAlert size={14} /> {warning}</div>
      {/each}
    {:else if validation.errors && validation.errors.length}
      {#each validation.errors as e, i (i)}
        <div class="error">{e.path ?? 'root'}: {e.message}</div>
      {/each}
    {:else}
      <div class="error">{validation.error}</div>
    {/if}
    {#if validation.valid && validation.plan}
      <!-- What a run of this definition, with these defaults, will do -
           the figure to expect before pressing Run, and the weights it
           would pull first, which the cost block never counts -->
      <ul
        class="plan"
        aria-label="what a run will do"
        title="from the workflow's own cost block and this server's model cache - a plan, not a promise"
      >
        {#each describePlan(validation.plan) as line, i (i)}
          <li class:warn={line.tone === 'warn'}>{line.text}</li>
        {/each}
      </ul>
    {/if}
  </div>
{/if}

{#if ed.view === 'flow'}
  <FlowView workflow={$state.snapshot(ed.doc)} onselect={focusStep} />
{:else}
  <EditorBody
    view={ed.view}
    jsonDraft={ed.jsonDraft}
    onjson={(raw) => ed.applyJson(raw)}
    hint="Schema-aware: completion, hover docs and validation come from the workflow schema. Changes apply when the editor loses focus."
    stickyTop="136px"
  >
    {#snippet form()}
      <div class="panel">
        <h2>Variables</h2>
        <VariablesForm
          mode="define"
          bind:variables={ed.doc.variables}
          idPrefix="wfvar-"
        />
      </div>

      {#if (ed.doc.steps ?? []).length > 1}
        <div class="densityrow">
          <span class="muted">steps</span>
          <span class="flex"></span>
          <button class="quiet" onclick={() => setAllModes('collapsed')}
            >collapse all</button
          >
          <button class="quiet" onclick={() => setAllModes('compact')}
            >compact all</button
          >
          <button class="quiet" onclick={() => setAllModes('full')}
            >expand all</button
          >
        </div>
      {/if}

      <div class="steps">
        {#each ed.doc.steps ?? [] as step, index (step)}
          <div
            id={'step-' + step.name}
            class="steprow"
            class:flowlit={hovered !== null && step.name === hovered}
          >
            <div class="railcell">
              <span
                class="ordinal"
                title={`step ${index + 1} of ${ed.doc.steps.length}`}
                >{index + 1}</span
              >
            </div>
            <StepEditor
              bind:step={ed.doc.steps[index]}
              {index}
              count={ed.doc.steps.length}
              references={stepReferences[index] ?? []}
              baseFolder={ed.folder === NEW_FOLDER ? '' : ed.folder}
              mode={modeOf(step)}
              flow={flow[index]}
              problems={problemsByStep.get(index) ?? []}
              onmodechange={(m) => setMode(step, m)}
              onhover={(n) => (hovered = n)}
              onremove={() => removeStep(index)}
              onmove={(delta) => moveStep(index, delta)}
            />
          </div>
        {/each}
      </div>

      <div class="addstep">
        <button
          class="quiet withicon"
          onclick={() => addStep('pipeline')}
          title="add a step that runs a diffusers pipeline"
        >
          <Plus size={14} />pipeline step
        </button>
        <button
          class="quiet withicon"
          onclick={() => addStep('task')}
          title="add a utility step - upscaling, segmentation, captioning, frame tools"
        >
          <Plus size={14} />task step
        </button>
        <button
          class="quiet withicon"
          onclick={() => addStep('workflow')}
          title="add a step that runs another workflow file with mapped arguments"
        >
          <Plus size={14} />sub-workflow step
        </button>
      </div>
    {/snippet}
  </EditorBody>
{/if}

<style>
  /* Save, Run and the way out stay reachable while a long workflow
     scrolls. Below 900px the app header wraps to a second row and its
     height stops being predictable, so the pinning is dropped there. */
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.6rem;
    position: sticky;
    top: 76px;
    z-index: 5;
    background: var(--bg);
    padding: 0.5rem 0;
    margin-bottom: 0.2rem;
    border-bottom: 1px solid var(--line);
  }
  @media (max-width: 900px) {
    .head {
      position: static;
    }
  }
  .crumbs {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.85rem;
    min-width: 0;
  }
  .crumbs .sep {
    opacity: 0.5;
  }
  .wfidwrap {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    max-width: 280px;
  }
  .wfidlabel {
    color: var(--muted);
    font-size: 0.75rem;
    text-transform: none;
  }
  /* Reads as a title until you reach for it, rather than as an unexplained
     text box sitting next to a breadcrumb */
  .wfid {
    font-weight: 700;
    background: transparent;
    border-color: transparent;
  }
  .wfid:hover {
    border-color: var(--line);
  }
  .wfid:focus {
    background: var(--panel-2);
  }
  .chip.unsaved {
    background: color-mix(in srgb, var(--warn) 22%, transparent);
    color: var(--warn);
  }
  .savebar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.6rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
    min-width: 0;
  }
  .pathchip {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    padding: 0.25rem 0.6rem;
    max-width: 100%;
  }
  .pathchip:hover {
    color: var(--ink);
    border-color: var(--accent);
  }
  .path {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .savebar .desc {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .filebar {
    margin-bottom: 1rem;
  }
  .filegrid {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    gap: 0.5rem 0.8rem;
    align-items: center;
  }
  .filegrid label {
    font-weight: 600;
    color: var(--muted);
    font-size: 0.85rem;
  }
  @container (max-width: 400px) {
    .filegrid {
      grid-template-columns: minmax(0, 1fr);
      gap: 0.2rem 0.5rem;
    }
  }
  .folderrow,
  .namerow {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
  }
  .filefoot {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.4rem 0.8rem;
    margin-top: 0.8rem;
  }
  .filefoot .path {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    min-width: 0;
  }
  .savename {
    max-width: 200px;
  }
  .panel {
    margin-bottom: 1rem;
  }
  .addstep {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem 0.6rem;
  }
  .densityrow {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: var(--space-2);
  }
  .densityrow button {
    font-size: 0.75rem;
    padding: 0.2rem 0.55rem;
  }
  .validation {
    position: relative;
    padding-right: 2.2rem;
  }
  .dismiss {
    position: absolute;
    top: var(--space-2);
    right: var(--space-2);
    border: 0;
    padding: 0.2rem 0.3rem;
  }
  /* Unsaved work makes Save the thing to do next, so it stops looking
     like the two quiet buttons beside it */
  .dirtybtn {
    border-color: var(--warn);
    color: var(--warn);
  }
  .steps {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    margin-bottom: var(--space-4);
  }
  .steprow {
    display: grid;
    grid-template-columns: 28px minmax(0, 1fr);
    gap: 0 var(--space-2);
  }
  .railcell {
    position: relative;
    display: flex;
    justify-content: center;
  }
  /* the connecting line - drawn per row so it spans the gaps too */
  .steprow:not(:last-child) .railcell::before {
    content: '';
    position: absolute;
    top: 26px;
    bottom: calc(-1 * var(--space-3));
    width: 2px;
    background: color-mix(in srgb, var(--accent) 35%, transparent);
  }
  .ordinal {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: var(--accent);
    color: var(--accent-ink);
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-top: var(--space-2);
    z-index: 1;
  }
  .steprow.flowlit :global(.panel.step) {
    border-color: var(--accent);
  }
  .ok {
    color: var(--good);
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
  }
  .warn {
    color: var(--warn);
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .error {
    color: var(--bad);
  }
  /* The plan under the verdict: figures the engine resolves, so mono, and
     quiet - a warn line is the one that changes the number to expect */
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
    display: inline;
  }
</style>
