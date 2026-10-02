<script lang="ts">
  import Suggest from '../ui/Suggest.svelte'
  import { editorLists } from '../editorLists.svelte'
  import {
    ChevronDown,
    ChevronRight,
    ChevronUp,
    Trash2,
    TriangleAlert,
  } from '@lucide/svelte'
  import ArgumentsEditor from './ArgumentsEditor.svelte'
  import PipelineOptions from './PipelineOptions.svelte'
  import StepDigestList from './StepDigestList.svelte'
  import MappingEditor from './MappingEditor.svelte'
  import { api } from '../api'
  import { stepDigest } from '../digest'
  import {
    contentTypeOptions,
    OFFLOAD_MODES,
    optionsWith,
    TORCH_DTYPES,
  } from '../editor'
  import type { StepFlow } from '../flow'

  let {
    step = $bindable(),
    index,
    count,
    references = [],
    baseFolder = '',
    mode = 'full',
    flow = undefined,
    problems = [],
    onmodechange = undefined,
    onhover = undefined,
    onremove,
    onmove,
  }: {
    step: Record<string, any>
    index: number
    count: number
    references?: string[]
    baseFolder?: string
    mode?: 'collapsed' | 'compact' | 'full'
    flow?: StepFlow
    problems?: string[]
    onmodechange?: (mode: 'collapsed' | 'compact' | 'full') => void
    onhover?: (stepName: string | null) => void
    onremove: () => void
    onmove: (delta: number) => void
  } = $props()

  const digest = $derived(stepDigest($state.snapshot(step)))

  // Mode is parent-owned (EditorPage persists it per step name) - every
  // internal change routes through the callback and flows back down
  function setModeInternal(next: 'collapsed' | 'compact' | 'full') {
    // A digest-click's section jump is the one case that should survive
    // the transition - openFull re-applies it right after this call
    openSection = ''
    onmodechange?.(next)
  }

  // The chevron re-opens to whichever expanded state the step last had
  let lastExpanded = $state<'compact' | 'full'>('compact')
  $effect(() => {
    if (mode !== 'collapsed') lastExpanded = mode
  })
  function toggleCollapsed() {
    setModeInternal(mode === 'collapsed' ? lastExpanded : 'collapsed')
  }

  // A compact line clicked open jumps straight to its section in full view
  let openSection = $state('')
  function openFull(section: string) {
    setModeInternal('full')
    openSection = section
  }

  const kind = $derived(
    step.pipeline
      ? 'pipeline'
      : step.task
        ? 'task'
        : step.workflow
          ? 'workflow'
          : step.pipeline_reference
            ? 'pipeline_reference'
            : 'unknown',
  )

  // pipeline_reference steps are edited as raw JSON - everything the
  // editor does not understand survives untouched
  let rawDraft = $state('')
  let rawError = $state('')

  function openRaw() {
    rawDraft = JSON.stringify(step[kind], null, 2)
  }

  function applyRaw() {
    try {
      step[kind] = JSON.parse(rawDraft)
      rawError = ''
    } catch (e) {
      rawError = e instanceof Error ? e.message : String(e)
    }
  }

  $effect(() => {
    if (kind !== 'pipeline' && kind !== 'task' && kind !== 'workflow') openRaw()
  })

  // A sub-workflow's argument suggestions are the target's own variables
  let workflowVariables = $state<Array<{ name: string; hint?: string }>>([])
  $effect(() => {
    workflowVariables = []
    const path = step.workflow?.path
    if (
      typeof path !== 'string' ||
      !path.endsWith('.json') ||
      path.includes(':')
    ) {
      return
    }
    // The engine resolves a step's relative path against the PARENT
    // workflow's directory, so the hint fetch must do the same
    const resolved =
      baseFolder && !path.startsWith('/') ? `${baseFolder}/${path}` : path
    const timer = setTimeout(() => {
      api
        .getWorkflow(resolved.slice(0, -'.json'.length))
        .then((fetched) => {
          workflowVariables = Object.entries(
            fetched.definition.variables ?? {},
          ).map(([name, value]) => ({
            name,
            hint: typeof value === 'string' ? value : JSON.stringify(value),
          }))
        })
        .catch(() => {})
    }, 300)
    return () => clearTimeout(timer)
  })

  const configuration = $derived(step.pipeline?.configuration ?? {})
  const pretrained = $derived(step.pipeline?.from_pretrained_arguments ?? {})
</script>

<div class="panel step">
  <div class="bar">
    <button
      class="quiet icon"
      onclick={toggleCollapsed}
      title={mode === 'collapsed' ? 'expand this step' : 'collapse this step'}
      aria-label={mode === 'collapsed'
        ? 'expand this step'
        : 'collapse this step'}
    >
      {#if mode === 'collapsed'}<ChevronRight size={15} />{:else}<ChevronDown
          size={15}
        />{/if}
    </button>
    <input
      class="name"
      bind:value={step.name}
      title="step name - how later steps reference this one"
    />
    <span class="kind muted">{kind}</span>
    {#each flow?.inputs ?? [] as producer (producer)}
      <span
        class="flowchip in"
        role="note"
        onmouseenter={() => onhover?.(producer)}
        onmouseleave={() => onhover?.(null)}
        title={`consumes previous_result:${producer}`}>← {producer}</span
      >
    {/each}
    {#each flow?.consumers ?? [] as consumer (consumer)}
      <span
        class="flowchip out"
        role="note"
        onmouseenter={() => onhover?.(consumer)}
        onmouseleave={() => onhover?.(null)}
        title={`step '${consumer}' consumes this step's result`}
        >→ {consumer}</span
      >
    {/each}
    {#if mode === 'collapsed'}
      <span class="muted summary" title={digest.summary}>{digest.summary}</span>
    {/if}
    <span class="flex"></span>
    {#if mode !== 'collapsed'}
      <div class="modeswitch" role="group" aria-label="step detail level">
        <button
          class="quiet"
          class:activebtn={mode === 'compact'}
          onclick={() => setModeInternal('compact')}
          title="one-line-per-area digest of what this step sets"
          >compact</button
        >
        <button
          class="quiet"
          class:activebtn={mode === 'full'}
          onclick={() => setModeInternal('full')}
          title="every field, editable">full</button
        >
      </div>
    {/if}
    <button
      class="quiet icon"
      disabled={index === 0}
      onclick={() => onmove(-1)}
      title="move up"
    >
      <ChevronUp size={15} />
    </button>
    <button
      class="quiet icon"
      disabled={index === count - 1}
      onclick={() => onmove(1)}
      title="move down"
    >
      <ChevronDown size={15} />
    </button>
    <button class="quiet icon" onclick={onremove} title="remove step">
      <Trash2 size={15} />
    </button>
  </div>

  {#each problems as problem (problem)}
    <div class="stepwarn"><TriangleAlert size={13} /> {problem}</div>
  {/each}
  {#if (flow?.resolvedRefs ?? 0) > 1}
    <div class="muted hint cartesian">
      {flow!.resolvedRefs} previous_result inputs - iterations multiply (every combination
      runs)
    </div>
  {/if}

  {#if mode === 'compact'}
    <StepDigestList lines={digest.lines} onopen={openFull} />
  {:else if mode === 'full'}
    {#if kind === 'pipeline'}
      <div class="grid">
        <label for={'ct-' + index}>pipeline</label>
        <Suggest
          id={'ct-' + index}
          suggestions={editorLists.pipelines}
          bind:value={configuration.component_type}
          placeholder="ZImagePipeline"
        />

        <label for={'model-' + index}>model</label>
        <input
          id={'model-' + index}
          bind:value={pretrained.model_name}
          placeholder="org/model-name or local path"
        />

        <label for={'dtype-' + index}>dtype</label>
        <select id={'dtype-' + index} bind:value={pretrained.torch_dtype}>
          {#each optionsWith(TORCH_DTYPES, pretrained.torch_dtype) as dtype (dtype)}<option
              >{dtype}</option
            >{/each}
        </select>

        <label for={'offload-' + index}>offload</label>
        <select
          id={'offload-' + index}
          value={configuration.offload ?? ''}
          onchange={(e) => {
            const v = e.currentTarget.value
            if (v) configuration.offload = v
            else delete configuration.offload
          }}
        >
          <option value="">none (resident)</option>
          {#each OFFLOAD_MODES as mode (mode)}<option value={mode}
              >{mode}</option
            >{/each}
        </select>

        <label for={'result-' + index}>save as</label>
        <select
          id={'result-' + index}
          value={step.result?.content_type ?? ''}
          onchange={(e) => {
            const v = e.currentTarget.value
            if (v) step.result = { ...(step.result ?? {}), content_type: v }
            else delete step.result
          }}
        >
          <option value="">don't save</option>
          {#each contentTypeOptions(step.result?.content_type) as contentType (contentType)}<option
              >{contentType}</option
            >{/each}
        </select>
      </div>

      <h3>arguments</h3>
      <ArgumentsEditor
        bind:args={step.pipeline.arguments}
        componentType={configuration.component_type ?? ''}
        suggestions={references}
      />

      <PipelineOptions
        bind:pipeline={step.pipeline}
        {index}
        {openSection}
        {references}
      />
    {:else if kind === 'task'}
      <div class="grid">
        <label for={'task-' + index}>command</label>
        <Suggest
          id={'task-' + index}
          suggestions={editorLists.taskCommands}
          bind:value={step.task.command}
          placeholder="e.g. upscale"
        />
      </div>
      <h3>arguments</h3>
      <ArgumentsEditor
        bind:args={step.task.arguments}
        componentType={step.task.command}
        target="task"
        suggestions={references}
      />
    {:else if kind === 'workflow'}
      <div class="grid">
        <label for={'wfpath-' + index}>path</label>
        <Suggest
          id={'wfpath-' + index}
          suggestions={editorLists.workflowFiles}
          bind:value={step.workflow.path}
          placeholder="Other.json, flux/FluxDev.json or builtin:h3_context_ir.json"
        />
      </div>
      <h3>arguments</h3>
      <MappingEditor
        bind:args={step.workflow.arguments}
        suggestions={workflowVariables}
        valueSuggestions={references}
      />
      <div class="muted hint">
        map the child's variables to values or references, e.g.
        previous_result:gen
      </div>
    {:else}
      <div class="raw">
        <textarea rows="8" bind:value={rawDraft} onchange={applyRaw}></textarea>
        {#if rawError}<div class="error">{rawError}</div>{/if}
        <div class="muted hint">
          {kind} steps are edited as JSON - changes apply on blur
        </div>
      </div>
    {/if}
  {/if}
</div>

<style>
  .bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.5rem;
  }
  .name {
    max-width: 220px;
    font-weight: 600;
  }
  .kind {
    font-size: 0.75rem;
  }
  .icon {
    display: inline-flex;
    align-items: center;
    padding: 0.3rem 0.45rem;
  }
  .grid {
    display: grid;
    grid-template-columns: 170px minmax(0, 480px);
    gap: 0.5rem 0.7rem;
    margin-top: 0.8rem;
    align-items: center;
  }
  @container (max-width: 400px) {
    .grid {
      grid-template-columns: minmax(0, 1fr);
      gap: 0.2rem;
    }
  }
  .grid label {
    font-weight: 600;
    color: var(--muted);
  }
  h3 {
    font-size: 0.8rem;
    text-transform: none;
    color: var(--muted);
    margin: 1rem 0 0.5rem;
  }
  .error {
    color: var(--bad);
    font-size: 0.8rem;
  }
  .raw {
    margin-top: 0.8rem;
  }
  .raw textarea {
    font-family: var(--font-mono);
    font-size: 0.82rem;
  }
  .error {
    color: var(--bad);
    font-size: 0.8rem;
    margin-top: 0.3rem;
  }
  .hint {
    font-size: 0.75rem;
    margin-top: 0.3rem;
  }
  .summary {
    font-size: 0.8rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
    flex: 1;
  }
  .modeswitch {
    display: inline-flex;
  }
  .modeswitch button {
    border-radius: 0;
    padding: 0.2rem 0.55rem;
    font-size: 0.75rem;
  }
  .modeswitch button:first-child {
    border-radius: var(--radius-1) 0 0 var(--radius-1);
  }
  .modeswitch button:last-child {
    border-radius: 0 var(--radius-1) var(--radius-1) 0;
    margin-left: -1px;
  }
  .activebtn {
    border-color: var(--accent);
    color: var(--accent);
    position: relative;
    z-index: 1;
  }
  .flowchip {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    padding: 0.05rem 0.5rem;
    border-radius: 999px;
    border: 1px solid;
    white-space: nowrap;
    cursor: default;
  }
  .flowchip.in {
    color: var(--accent);
    border-color: color-mix(in srgb, var(--accent) 45%, transparent);
    background: color-mix(in srgb, var(--accent) 12%, transparent);
  }
  .flowchip.out {
    color: var(--good);
    border-color: color-mix(in srgb, var(--good) 45%, transparent);
    background: color-mix(in srgb, var(--good) 12%, transparent);
  }
  .stepwarn {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--warn);
    font-size: 0.85rem;
    margin-top: var(--space-2);
  }
  .cartesian {
    margin-top: var(--space-1);
  }
</style>
