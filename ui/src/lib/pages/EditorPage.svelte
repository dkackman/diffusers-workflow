<script lang="ts">
  import {
    CircleCheck,
    FileJson,
    Play,
    Plus,
    Save,
    TriangleAlert,
  } from 'lucide-svelte'
  import { api } from '../api'
  import { go } from '../router.svelte'
  import { emptyWorkflow, emptyStep } from '../editor'
  import StepEditor from '../editor/StepEditor.svelte'
  import JsonEditor from '../editor/JsonEditor.svelte'
  import type { ValidationResult, WorkflowDefinition } from '../types'

  let { name = '' }: { name?: string } = $props()

  let workflow = $state<Record<string, any>>(emptyWorkflow())
  let saveName = $state('')
  let workflowDir = $state('')
  let pipelines = $state<string[]>([])
  let modelClasses = $state<string[]>([])
  let schedulerClasses = $state<string[]>([])
  let quantizationClasses = $state<string[]>([])
  let validation = $state<ValidationResult | null>(null)
  let status = $state('')
  let error = $state('')
  let showJson = $state(false)
  let jsonDraft = $state('')
  let busy = $state(false)

  $effect(() => {
    api.listPipelines().then((r) => (pipelines = r.pipelines))
    api.listClasses('models').then((r) => (modelClasses = r.classes))
    api.listClasses('schedulers').then((r) => (schedulerClasses = r.classes))
    api.listClasses('quantization').then((r) => (quantizationClasses = r.classes))
    api.listWorkflows().then((r) => (workflowDir = r.workflow_dir))
    validation = null
    error = ''
    if (name) {
      saveName = name
      api
        .getWorkflow(name)
        .then((definition) => (workflow = definition as WorkflowDefinition))
        .catch((e) => (error = e.message))
    } else {
      // A gallery "open as workflow" hands the definition over in
      // sessionStorage - one-shot, so a plain "New" stays a blank slate
      const imported = sessionStorage.getItem('dw-editor-import')
      if (imported) {
        sessionStorage.removeItem('dw-editor-import')
        try {
          workflow = JSON.parse(imported)
          status = 'Imported from image metadata'
        } catch {
          workflow = emptyWorkflow()
        }
      } else {
        workflow = emptyWorkflow()
      }
      saveName = ''
    }
  })

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

  function addStep() {
    workflow.steps = [...(workflow.steps ?? []), emptyStep()]
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
      validation = await api.validate($state.snapshot(workflow) as WorkflowDefinition)
      return validation.valid
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
      return false
    } finally {
      busy = false
    }
  }

  async function save() {
    if (!saveName) {
      status = 'Give the workflow a file name first'
      return
    }
    if (!(await validate())) return
    busy = true
    try {
      const result = await api.saveWorkflow(
        saveName,
        $state.snapshot(workflow) as WorkflowDefinition,
      )
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
      const job = await api.submitJob({
        workflow: $state.snapshot(workflow) as WorkflowDefinition,
      })
      go('jobs', job.id)
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
    } finally {
      busy = false
    }
  }

  function toggleJson() {
    if (!showJson) jsonDraft = JSON.stringify($state.snapshot(workflow), null, 2)
    showJson = !showJson
  }

  function applyJson(raw: string) {
    jsonDraft = raw
    try {
      workflow = JSON.parse(raw)
      error = ''
    } catch (e) {
      error = `JSON: ${e instanceof Error ? e.message : e}`
    }
  }
</script>

<datalist id="pipeline-classes">
  {#each pipelines as pipeline}<option value={pipeline}></option>{/each}
</datalist>
<datalist id="model-classes">
  {#each modelClasses as model}<option value={model}></option>{/each}
</datalist>
<datalist id="scheduler-classes">
  {#each schedulerClasses as scheduler}<option value={scheduler}></option>{/each}
</datalist>
<datalist id="quantization-classes">
  {#each quantizationClasses as quantization}<option value={quantization}></option>{/each}
</datalist>

<div class="head">
  <a href="#/workflows" class="muted">← workflows</a>
  <input class="wfid" bind:value={workflow.id} title="workflow id" />
  <span class="flex"></span>
  <button class="quiet withicon" onclick={toggleJson}>
    <FileJson size={14} />{showJson ? 'form' : 'JSON'}
  </button>
  <button class="quiet withicon" onclick={validate} disabled={busy}>
    <CircleCheck size={14} />Validate
  </button>
  <button class="quiet withicon" onclick={save} disabled={busy}>
    <Save size={14} />Save
  </button>
  <button class="withicon" onclick={run} disabled={busy}>
    <Play size={14} />Run
  </button>
</div>

<div class="savebar muted">
  saving as <input class="savename" bind:value={saveName} placeholder="MyWorkflow" />
  <span>.json in {workflowDir}</span>
</div>

{#if error}<p class="error">{error}</p>{/if}
{#if status}<p class="muted">{status}</p>{/if}

{#if validation}
  <div class="panel validation" class:bad={!validation.valid}>
    {#if validation.valid && !validation.warnings.length}
      <span class="ok"><CircleCheck size={14} /> schema-valid, no argument warnings</span>
    {:else if validation.valid}
      {#each validation.warnings as warning}
        <div class="warn"><TriangleAlert size={14} /> {warning}</div>
      {/each}
    {:else}
      <div class="error">{validation.error}</div>
    {/if}
  </div>
{/if}

{#if showJson}
  <JsonEditor value={jsonDraft} onchange={applyJson} height="560px" />
  <p class="muted hint">
    Schema-aware: completion, hover docs and validation come from the workflow
    schema. Changes apply when the editor loses focus.
  </p>
{:else}
  <div class="panel">
    <h2>Variables</h2>
    <div class="vars">
      {#each variables as key (key)}
        <label for={'wfvar-' + key}>{key}</label>
        <input id={'wfvar-' + key} bind:value={workflow.variables[key]} />
        <button class="quiet icon" onclick={() => removeVariable(key)} title="remove">
          ×
        </button>
      {/each}
      <input placeholder="new variable name…" bind:value={newVariable} />
      <button class="quiet withicon addvar" onclick={addVariable} disabled={!newVariable}>
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
      {pipelines}
      onremove={() => removeStep(index)}
      onmove={(delta) => moveStep(index, delta)}
    />
  {/each}

  <button class="quiet withicon" onclick={addStep}><Plus size={14} />add step</button>
{/if}

<style>
  .head { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem; }
  .wfid { max-width: 240px; font-weight: 700; }
  .flex { flex: 1; }
  .withicon { display: inline-flex; align-items: center; gap: 0.35rem; }
  .savebar { display: flex; align-items: center; gap: 0.4rem; margin-bottom: 1rem; font-size: 0.85rem; }
  .savename { max-width: 200px; }
  .panel { margin-bottom: 1rem; }
  .vars {
    display: grid; grid-template-columns: minmax(140px, auto) 1fr auto;
    gap: 0.5rem 0.8rem; align-items: center;
  }
  .vars label { font-weight: 600; color: var(--muted); }
  .icon { padding: 0.3rem 0.55rem; }
  .addvar { justify-self: start; }
  .validation.bad { border-color: var(--bad); }
  .ok { color: var(--good); display: inline-flex; align-items: center; gap: 0.4rem; }
  .warn { color: var(--warn); display: flex; align-items: center; gap: 0.4rem; }
  .error { color: var(--bad); }
  .hint { font-size: 0.8rem; }
</style>
