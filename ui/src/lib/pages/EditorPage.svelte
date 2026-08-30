<script lang="ts">
  import {
    CircleCheck,
    Columns2,
    Braces,
    LayoutList,
    Play,
    Plus,
    Save,
    TriangleAlert,
  } from '@lucide/svelte'
  import { api } from '../api'
  import { go } from '../router.svelte'
  import {
    coerce,
    danglingReferences,
    emptyWorkflow,
    emptyStep,
    emptyTaskStep,
    emptyWorkflowStep,
    isReference,
    referenceSuggestions,
    widgetFor,
  } from '../editor'
  import { loadPromptLibrary, promptLibrary } from '../promptlib.svelte'
  import { PROMPT_LIST_ID, promptListId, promptTooltip } from '../prompts'
  import StepEditor from '../editor/StepEditor.svelte'
  import JsonEditor from '../editor/JsonEditor.svelte'
  import type { ValidationResult, WorkflowDefinition } from '../types'

  let { name = '' }: { name?: string } = $props()

  let workflow = $state<Record<string, any>>(emptyWorkflow())
  // Raw text as typed, per variable - the committed value only updates on
  // blur, and the prompt datalist should attach as soon as prompt: is typed
  let varDrafts = $state<Record<string, string>>({})
  let saveName = $state('')
  let workflowDir = $state('')
  let pipelines = $state<string[]>([])
  let modelClasses = $state<string[]>([])
  let schedulerClasses = $state<string[]>([])
  let quantizationClasses = $state<string[]>([])
  let taskCommands = $state<string[]>([])
  let workflowFiles = $state<string[]>([])
  let folder = $state('')
  let newFolder = $state('')

  // Existing folders, from the listing - one level is the designed depth,
  // but any deeper directories that exist still appear and keep working
  const folders = $derived(
    [
      ...new Set(
        workflowFiles
          .filter((file) => file.includes('/'))
          .map((file) => file.split('/').slice(0, -1).join('/')),
      ),
    ].sort(),
  )
  let validation = $state<ValidationResult | null>(null)
  let status = $state('')
  let error = $state('')
  type EditorView = 'form' | 'split' | 'json'
  let view = $state<EditorView>(
    (() => {
      try {
        const stored = localStorage.getItem('dw-editor-view')
        if (stored === 'form' || stored === 'split' || stored === 'json')
          return stored
        // migrate the old boolean split preference
        return localStorage.getItem('dw-editor-split') === '1'
          ? 'split'
          : 'form'
      } catch {
        return 'form'
      }
    })(),
  )

  function setView(next: EditorView) {
    view = next
    try {
      localStorage.setItem('dw-editor-view', next)
    } catch {
      /* session only */
    }
  }

  let jsonDraft = $state('')
  let jsonParseFailed = $state(false)

  // Mirror the workflow into the JSON surfaces. A failed parse pins the
  // raw text so the user's broken edit isn't regenerated out from under
  // them before they can fix it.
  $effect(() => {
    const pretty = JSON.stringify($state.snapshot(workflow), null, 2)
    if (!jsonParseFailed) jsonDraft = pretty
  })
  let busy = $state(false)
  let baseline = $state('')

  const serialized = $derived(JSON.stringify($state.snapshot(workflow)))
  const dirty = $derived(baseline !== '' && serialized !== baseline)
  // promptLibrary.names stays undefined until the listing lands - a
  // missing library must not flag every prompt: reference as dangling
  const referenceProblems = $derived(
    danglingReferences($state.snapshot(workflow), promptLibrary.names),
  )
  // Memoized so datalist options keep stable DOM identity - churn on every
  // render made the browser's suggestion dropdown flaky on first focus
  const stepReferences = $derived.by(() => {
    const snapshot = $state.snapshot(workflow)
    return ((snapshot.steps as unknown[]) ?? []).map((_, index) =>
      referenceSuggestions(snapshot, index, promptLibrary.names ?? []),
    )
  })

  $effect(() => {
    api.listPipelines().then((r) => (pipelines = r.pipelines))
    api.listClasses('models').then((r) => (modelClasses = r.classes))
    api.listClasses('schedulers').then((r) => (schedulerClasses = r.classes))
    api
      .listClasses('quantization')
      .then((r) => (quantizationClasses = r.classes))
    api
      .listTasks()
      .then(
        (r) =>
          (taskCommands = [
            ...r.commands,
            ...r.image_processors,
            ...r.video_processors,
          ].sort()),
      )
    api.listWorkflows().then((r) => {
      workflowFiles = r.workflows.map((file) => `${file}.json`)
      workflowDir = r.workflow_dir
    })
    loadPromptLibrary()
    validation = null
    error = ''
    if (name) {
      const segments = name.split('/')
      saveName = segments[segments.length - 1]
      folder = segments.slice(0, -1).join('/')
      api
        .getWorkflow(name)
        .then((definition) => {
          workflow = definition as WorkflowDefinition
          baseline = JSON.stringify(definition)
        })
        .catch((e) => (error = e.message))
    } else {
      // A gallery "open as workflow" hands the definition over in
      // sessionStorage - one-shot, so a plain "New" stays a blank slate
      folder = sessionStorage.getItem('dw-editor-folder') ?? ''
      sessionStorage.removeItem('dw-editor-folder')
      const imported = sessionStorage.getItem('dw-editor-import')
      // Built as a plain local object and assigned once: reading the
      // workflow proxy here would subscribe this effect to every edit and
      // re-fire all the listing calls on each keystroke
      let fresh = emptyWorkflow() as Record<string, any>
      if (imported) {
        sessionStorage.removeItem('dw-editor-import')
        try {
          fresh = JSON.parse(imported)
          status = 'Imported from image metadata'
        } catch {
          /* unreadable hand-off - stay with the blank slate */
        }
      }
      workflow = fresh
      baseline = JSON.stringify(fresh)
      saveName = ''
    }
  })

  // Unsaved edits should survive an accidental tab close. dirty is read
  // inside the handler only, so the listener registers exactly once
  $effect(() => {
    const guard = (event: BeforeUnloadEvent) => {
      if (dirty) event.preventDefault()
    }
    window.addEventListener('beforeunload', guard)
    return () => {
      window.removeEventListener('beforeunload', guard)
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

  function setVariable(key: string, raw: string) {
    // Coerce to the default's type - schema validation runs on the
    // definition itself, so "9" where 9 belongs breaks the workflow
    workflow.variables[key] = coerce(
      widgetFor(undefined, workflow.variables[key]),
      raw,
    )
  }

  const variables = $derived(Object.keys(workflow.variables ?? {}))
  let newVariable = $state('')

  function addVariable() {
    if (!newVariable) return
    workflow.variables = workflow.variables ?? {}
    workflow.variables[newVariable] = ''
    newVariable = ''
  }

  function removeVariable(key: string) {
    delete workflow.variables[key]
  }

  function addStep(kind: string) {
    const step =
      kind === 'task'
        ? emptyTaskStep()
        : kind === 'workflow'
          ? emptyWorkflowStep()
          : emptyStep()
    workflow.steps = [...(workflow.steps ?? []), step]
  }

  function removeStep(index: number) {
    workflow.steps.splice(index, 1)
  }

  function moveStep(index: number, delta: number) {
    const steps = workflow.steps
    const target = index + delta
    if (target < 0 || target >= steps.length) return
    ;[steps[index], steps[target]] = [steps[target], steps[index]]
  }

  async function validate(): Promise<boolean> {
    busy = true
    status = ''
    try {
      validation = await api.validate(
        $state.snapshot(workflow) as WorkflowDefinition,
      )
      return validation.valid
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
      return false
    } finally {
      busy = false
    }
  }

  function savePath(): string | null {
    if (!saveName) return null
    const directory = folder === '__new__' ? newFolder.trim() : folder
    if (folder === '__new__' && !/^[\w][\w.-]*$/.test(directory)) return null
    return directory ? `${directory}/${saveName}` : saveName
  }

  async function save() {
    // Validation failures are errors (red), not statuses (green checkmark)
    status = ''
    const path = savePath()
    if (!path) {
      if (!saveName) error = 'Give the workflow a file name first'
      else if (!newFolder.trim()) error = 'Name the new folder first'
      else error = 'Folder names: letters, numbers, dot, dash, underscore'
      return
    }
    if (!(await validate())) return
    busy = true
    try {
      const result = await api.saveWorkflow(
        path,
        $state.snapshot(workflow) as WorkflowDefinition,
      )
      if (folder === '__new__') {
        folder = newFolder.trim()
        newFolder = ''
      }
      // The folder picker's options come from the listing - a folder this
      // save just created must appear there, or the select falls back to
      // "(root)" while the state still names the folder
      if (!workflowFiles.includes(`${path}.json`)) {
        workflowFiles = [...workflowFiles, `${path}.json`]
      }
      baseline = JSON.stringify($state.snapshot(workflow))
      status = `Saved to ${result.path}`
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
    } finally {
      busy = false
    }
  }

  async function run() {
    if (!(await validate())) return
    busy = true
    try {
      // base_dir anchors relative paths (images, sub-workflow files) the
      // way running the saved file would - at the workflow's own folder
      const directory = folder && folder !== '__new__' ? `/${folder}` : ''
      const job = await api.submitJob({
        workflow: $state.snapshot(workflow) as WorkflowDefinition,
        base_dir: `${workflowDir}${directory}`,
      })
      go('jobs', job.id)
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
    } finally {
      busy = false
    }
  }

  function applyJson(raw: string) {
    jsonDraft = raw
    try {
      workflow = JSON.parse(raw)
      jsonParseFailed = false
      error = ''
    } catch (e) {
      jsonParseFailed = true
      error = `JSON: ${e instanceof Error ? e.message : e}`
    }
  }
</script>

<datalist id="pipeline-classes">
  {#each pipelines as pipeline (pipeline)}<option value={pipeline}
    ></option>{/each}
</datalist>
<datalist id="model-classes">
  {#each modelClasses as model (model)}<option value={model}></option>{/each}
</datalist>
<datalist id="scheduler-classes">
  {#each schedulerClasses as scheduler (scheduler)}<option value={scheduler}
    ></option>{/each}
</datalist>
<datalist id="quantization-classes">
  {#each quantizationClasses as quantization (quantization)}<option
      value={quantization}
    ></option>{/each}
</datalist>
<datalist id="task-commands">
  {#each taskCommands as command (command)}<option value={command}
    ></option>{/each}
</datalist>
<svelte:window onkeydown={onKeydown} />

<datalist id="workflow-files">
  {#each workflowFiles as file (file)}<option value={file}></option>{/each}
</datalist>

<datalist id={PROMPT_LIST_ID}>
  {#each promptLibrary.names ?? [] as promptName (promptName)}<option
      value={'prompt:' + promptName}
    ></option>{/each}
</datalist>

<div class="head">
  <a href="#/workflows" class="muted">← workflows</a>
  <input class="wfid" bind:value={workflow.id} title="workflow id" />
  <span class="flex"></span>
  <div class="viewswitch" role="group" aria-label="editor view">
    <button
      class="quiet withicon"
      class:activebtn={view === 'form'}
      onclick={() => setView('form')}
      title="edit with introspection-driven forms"
    >
      <LayoutList size={14} />form
    </button>
    <button
      class="quiet withicon"
      class:activebtn={view === 'split'}
      onclick={() => setView('split')}
      title="form beside the JSON - both editable, blur applies"
    >
      <Columns2 size={14} />split
    </button>
    <button
      class="quiet withicon"
      class:activebtn={view === 'json'}
      onclick={() => setView('json')}
      title="edit the raw JSON, schema-aware"
    >
      <Braces size={14} />JSON
    </button>
  </div>
  <button
    class="quiet withicon"
    onclick={validate}
    disabled={busy}
    title="check against the schema and real pipeline signatures, without running"
  >
    <CircleCheck size={14} />Validate
  </button>
  <button
    class="quiet withicon"
    class:dirtybtn={dirty}
    onclick={save}
    disabled={busy}
    title="validate, then write to the workflow directory under the name below (Ctrl+S)"
  >
    <Save size={14} />Save{#if dirty}<span class="dirtydot"></span>{/if}
  </button>
  <button
    class="withicon"
    onclick={run}
    disabled={busy}
    title="validate, then queue this definition as a job - no save needed (Ctrl+Enter)"
  >
    <Play size={14} />Run
  </button>
</div>

<div class="savebar muted">
  saving as
  <select class="folderpick" bind:value={folder} title="folder to save into">
    <option value="">(root)</option>
    {#each folders as existing (existing)}<option value={existing}
        >{existing}/</option
      >{/each}
    <option value="__new__">new folder…</option>
  </select>
  {#if folder === '__new__'}
    <input
      class="newfolder"
      bind:value={newFolder}
      placeholder="folder name"
      title="name for the new folder at the root of the workflow directory"
    />
    <span>/</span>
  {/if}
  <input class="savename" bind:value={saveName} placeholder="MyWorkflow" />
  <span>.json in {workflowDir}</span>
  <input
    class="descfield"
    spellcheck="true"
    value={workflow.description ?? ''}
    placeholder="description - shown on the workflow card"
    title="a short description of what this workflow does"
    onchange={(e) => {
      const v = e.currentTarget.value
      if (v) workflow.description = v
      else delete workflow.description
    }}
  />
</div>

{#if error}<p class="error">{error}</p>{/if}
{#if referenceProblems.length}
  <div class="panel warn-edge refproblems">
    {#each referenceProblems as problem, i (i)}
      <div><TriangleAlert size={13} /> {problem}</div>
    {/each}
  </div>
{/if}
{#if status}
  <p class="status"><CircleCheck size={14} />{status}</p>
{/if}

{#if validation}
  <div
    class="panel validation"
    class:error-edge={!validation.valid}
    class:warn-edge={validation.valid && validation.warnings.length > 0}
    class:good-edge={validation.valid && validation.warnings.length === 0}
  >
    {#if validation.valid && !validation.warnings.length}
      <span class="ok"
        ><CircleCheck size={14} /> schema-valid, no argument warnings</span
      >
    {:else if validation.valid}
      {#each validation.warnings as warning, i (i)}
        <div class="warn"><TriangleAlert size={14} /> {warning}</div>
      {/each}
    {:else}
      <div class="error">{validation.error}</div>
    {/if}
  </div>
{/if}

{#if view === 'json'}
  <JsonEditor value={jsonDraft} onchange={applyJson} height="560px" />
  <p class="muted hint">
    Schema-aware: completion, hover docs and validation come from the workflow
    schema. Changes apply when the editor loses focus.
  </p>
{:else}
  <div class="editwrap" class:splitcols={view === 'split'}>
    <div class="formcol">
      <div class="panel">
        <h2>Variables</h2>
        <div class="vars">
          {#each variables as key (key)}
            <label for={'wfvar-' + key}>{key}</label>
            <input
              id={'wfvar-' + key}
              class:ref={isReference(workflow.variables[key])}
              list={promptListId(varDrafts[key] ?? workflow.variables[key])}
              autocomplete="off"
              title={promptTooltip(
                workflow.variables[key],
                promptLibrary.texts,
              )}
              value={typeof workflow.variables[key] === 'object'
                ? JSON.stringify(workflow.variables[key])
                : String(workflow.variables[key] ?? '')}
              oninput={(e) => (varDrafts[key] = e.currentTarget.value)}
              onchange={(e) => setVariable(key, e.currentTarget.value)}
            />
            <button
              class="quiet icon"
              onclick={() => removeVariable(key)}
              title="remove this variable"
              aria-label="remove this variable"
            >
              ×
            </button>
          {/each}
          <input placeholder="new variable name…" bind:value={newVariable} />
          <button
            class="quiet withicon addvar"
            onclick={addVariable}
            disabled={!newVariable}
            title="add this variable to the workflow"
          >
            <Plus size={14} />add
          </button>
          <span></span>
        </div>
      </div>

      {#each workflow.steps ?? [] as step, index (step)}
        <StepEditor
          bind:step={workflow.steps[index]}
          {index}
          count={workflow.steps.length}
          references={stepReferences[index] ?? []}
          baseFolder={folder === '__new__' ? '' : folder}
          onremove={() => removeStep(index)}
          onmove={(delta) => moveStep(index, delta)}
        />
      {/each}

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
    </div>
    {#if view === 'split'}
      <div class="jsoncol">
        <JsonEditor
          value={jsonDraft}
          onchange={applyJson}
          height="calc(100vh - 200px)"
        />
      </div>
    {/if}
  </div>
{/if}

<style>
  .head {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.4rem;
  }
  .wfid {
    max-width: 240px;
    font-weight: 700;
  }
  .flex {
    flex: 1;
  }
  .withicon {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  .savebar {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
  }
  .savename {
    max-width: 200px;
  }
  .folderpick {
    max-width: 160px;
  }
  .newfolder {
    max-width: 140px;
  }
  .descfield {
    flex: 1;
    min-width: 220px;
  }
  .panel {
    margin-bottom: 1rem;
  }
  .vars {
    display: grid;
    grid-template-columns: minmax(140px, auto) 1fr auto;
    gap: 0.5rem 0.8rem;
    align-items: center;
  }
  .vars label {
    font-weight: 600;
    color: var(--muted);
  }
  .icon {
    padding: 0.3rem 0.55rem;
  }
  .addvar {
    justify-self: start;
  }
  .addstep {
    display: flex;
    gap: 0.6rem;
  }
  .editwrap.splitcols {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(360px, 44%);
    gap: 1.1rem;
    align-items: start;
  }
  .jsoncol {
    position: sticky;
    top: 66px;
  }
  @media (max-width: 1100px) {
    .editwrap.splitcols {
      grid-template-columns: 1fr;
    }
    .jsoncol {
      position: static;
    }
  }
  .viewswitch {
    display: inline-flex;
  }
  .viewswitch button {
    border-radius: 0;
  }
  .viewswitch button:first-child {
    border-radius: 6px 0 0 6px;
  }
  .viewswitch button:last-child {
    border-radius: 0 6px 6px 0;
  }
  .viewswitch button + button {
    margin-left: -1px;
  }
  .activebtn {
    border-color: var(--accent);
    color: var(--accent);
    position: relative;
    z-index: 1;
  }
  .status {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--good);
    font-size: 0.9rem;
  }
  .dirtydot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--warn);
    margin-left: 0.15rem;
  }
  .refproblems {
    color: var(--warn);
    font-size: 0.9rem;
  }
  .refproblems div {
    display: flex;
    align-items: center;
    gap: 0.4rem;
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
  .hint {
    font-size: 0.8rem;
  }
</style>
