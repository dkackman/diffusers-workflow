<script lang="ts">
  import {
    Boxes,
    ChevronDown,
    ChevronRight,
    ChevronUp,
    Layers,
    Timer,
    Trash2,
    Zap,
  } from '@lucide/svelte'
  import ArgumentsEditor from './ArgumentsEditor.svelte'
  import ComponentEditor from './ComponentEditor.svelte'
  import LorasEditor from './LorasEditor.svelte'
  import MappingEditor from './MappingEditor.svelte'
  import { api } from '../api'
  import { stepDigest } from '../digest'
  import {
    ATTENTION_BACKENDS,
    CACHE_TYPES,
    COMPONENT_SLOTS,
    CONTENT_TYPES,
    TORCH_DTYPES,
    classDescription,
    emptyComponent,
    setNumber,
  } from '../editor'

  let {
    step = $bindable(),
    index,
    count,
    references = [],
    baseFolder = '',
    mode = 'full',
    onmodechange = undefined,
    onremove,
    onmove,
  }: {
    step: Record<string, any>
    index: number
    count: number
    references?: string[]
    baseFolder?: string
    mode?: 'collapsed' | 'compact' | 'full'
    onmodechange?: (mode: 'collapsed' | 'compact' | 'full') => void
    onremove: () => void
    onmove: (delta: number) => void
  } = $props()

  const referenceListId = $derived(`refs-${index}`)

  const digest = $derived(stepDigest($state.snapshot(step)))

  // Mode is parent-owned (EditorPage persists it per step name) - every
  // internal change routes through the callback and flows back down
  function setModeInternal(next: 'collapsed' | 'compact' | 'full') {
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
    openSection = section
    setModeInternal('full')
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
        .then((definition) => {
          workflowVariables = Object.entries(definition.variables ?? {}).map(
            ([name, value]) => ({
              name,
              hint: typeof value === 'string' ? value : JSON.stringify(value),
            }),
          )
        })
        .catch(() => {})
    }, 300)
    return () => clearTimeout(timer)
  })

  const configuration = $derived(step.pipeline?.configuration ?? {})
  const pretrained = $derived(step.pipeline?.from_pretrained_arguments ?? {})

  // ---- optional blocks: components, loras, scheduler, acceleration ----

  const activeSlots = $derived(
    COMPONENT_SLOTS.filter((slot) => step.pipeline && slot in step.pipeline),
  )
  let addSlot = $state('')

  function addComponent() {
    if (!addSlot) return
    step.pipeline[addSlot] = emptyComponent()
    addSlot = ''
  }

  function toggleScheduler(enabled: boolean) {
    if (enabled) {
      step.pipeline.scheduler = step.pipeline.scheduler ?? {
        configuration: { scheduler_type: '' },
        from_config_args: {},
      }
    } else {
      delete step.pipeline.scheduler
    }
  }

  function toggleCache(enabled: boolean) {
    if (enabled) {
      configuration.cache = configuration.cache ?? {
        type: 'first_block',
        threshold: 0.1,
      }
    } else {
      delete configuration.cache
    }
  }

  let compatibles = $state<string[]>([])
  $effect(() => {
    compatibles = []
    const schedulerType =
      step.pipeline?.scheduler?.configuration?.scheduler_type
    // Only complete-looking names - prefixes typed on the way to a real one
    // would each fire a doomed lookup
    if (!schedulerType || !schedulerType.endsWith('Scheduler')) return
    const timer = setTimeout(() => {
      classDescription(schedulerType, 'init').then(
        (d) => (compatibles = d?.compatibles ?? []),
      )
    }, 300)
    return () => clearTimeout(timer)
  })
</script>

<datalist id={referenceListId}>
  {#each references as reference (reference)}<option value={reference}
    ></option>{/each}
</datalist>

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

  {#if mode === 'compact'}
    <div class="digest">
      {#each digest.lines as line (line.section)}
        <button
          class="digestline"
          onclick={() => openFull(line.section)}
          title="edit in full view"
        >
          <span class="digestsection muted">{line.section}</span>
          <span class="digesttext">{line.text}</span>
        </button>
      {:else}
        <div class="muted hint">nothing set yet - switch to full to edit</div>
      {/each}
    </div>
  {:else if mode === 'full'}
    {#if kind === 'pipeline'}
      <div class="grid">
        <label for={'ct-' + index}>pipeline</label>
        <input
          id={'ct-' + index}
          list="pipeline-classes"
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
          {#each TORCH_DTYPES as dtype (dtype)}<option>{dtype}</option>{/each}
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
          <option value="model">model</option>
          <option value="sequential">sequential</option>
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
          {#each CONTENT_TYPES as contentType (contentType)}<option
              >{contentType}</option
            >{/each}
        </select>
      </div>

      <h3>arguments</h3>
      <ArgumentsEditor
        bind:args={step.pipeline.arguments}
        componentType={configuration.component_type ?? ''}
        listId={referenceListId}
      />

      <details open={activeSlots.length > 0 || openSection === 'components'}>
        <summary
          ><Boxes size={13} /> components
          <span class="muted">({activeSlots.length})</span></summary
        >
        <div class="section">
          {#each activeSlots as slot (slot)}
            <ComponentEditor
              {slot}
              bind:component={step.pipeline[slot]}
              listId={referenceListId}
              onremove={() => delete step.pipeline[slot]}
            />
          {/each}
          <div class="addrow">
            <select bind:value={addSlot}>
              <option value="">add component…</option>
              {#each COMPONENT_SLOTS.filter((slot) => !activeSlots.includes(slot)) as slot (slot)}
                <option value={slot}>{slot}</option>
              {/each}
            </select>
            <button
              class="quiet"
              onclick={addComponent}
              disabled={!addSlot}
              title="declare the selected component on this pipeline"
              >add</button
            >
          </div>
        </div>
      </details>

      <details
        open={(step.pipeline.loras ?? []).length > 0 || openSection === 'loras'}
      >
        <summary
          ><Layers size={13} /> LoRAs
          <span class="muted">({(step.pipeline.loras ?? []).length})</span
          ></summary
        >
        <div class="section">
          <LorasEditor bind:pipeline={step.pipeline} />
        </div>
      </details>

      <details open={!!step.pipeline.scheduler || openSection === 'scheduler'}>
        <summary><Timer size={13} /> scheduler</summary>
        <div class="section grid2">
          <label for={'sched-' + index}>replace scheduler</label>
          <input
            id={'sched-' + index}
            type="checkbox"
            class="check"
            checked={!!step.pipeline.scheduler}
            onchange={(e) => toggleScheduler(e.currentTarget.checked)}
          />
          {#if step.pipeline.scheduler}
            <label for={'schedtype-' + index}>scheduler_type</label>
            <div>
              <input
                id={'schedtype-' + index}
                list="scheduler-classes"
                bind:value={
                  step.pipeline.scheduler.configuration.scheduler_type
                }
                placeholder="e.g. FlowMatchEulerDiscreteScheduler"
              />
              {#if compatibles.length}
                <div class="muted hint">
                  interchangeable with: {compatibles
                    .slice(0, 6)
                    .join(', ')}{compatibles.length > 6 ? ', …' : ''}
                </div>
              {/if}
            </div>
            <label for={'schedargs-' + index}>from_config_args</label>
            <ArgumentsEditor
              bind:args={step.pipeline.scheduler.from_config_args}
              componentType={step.pipeline.scheduler.configuration
                .scheduler_type ?? ''}
              target="init"
              listId={referenceListId}
            />
          {/if}
        </div>
      </details>

      <details
        open={!!configuration.cache ||
          !!configuration.attention_backend ||
          openSection === 'acceleration'}
      >
        <summary><Zap size={13} /> acceleration</summary>
        <div class="section grid2">
          <label for={'cache-' + index}>cache</label>
          <div class="inline-row">
            <input
              id={'cache-' + index}
              type="checkbox"
              class="check"
              checked={!!configuration.cache}
              onchange={(e) => toggleCache(e.currentTarget.checked)}
            />
            {#if configuration.cache}
              <select bind:value={configuration.cache.type}>
                {#each CACHE_TYPES as cacheType (cacheType)}<option
                    >{cacheType}</option
                  >{/each}
              </select>
              <input
                class="num"
                placeholder="threshold"
                value={configuration.cache.threshold ?? ''}
                onchange={(e) =>
                  setNumber(
                    configuration.cache,
                    'threshold',
                    e.currentTarget.value,
                  )}
              />
            {/if}
          </div>

          <label for={'attn-' + index}>attention backend</label>
          <input
            id={'attn-' + index}
            list="attention-backends"
            value={configuration.attention_backend ?? ''}
            placeholder="pipeline default"
            onchange={(e) => {
              const v = e.currentTarget.value
              if (v) configuration.attention_backend = v
              else delete configuration.attention_backend
            }}
          />
          <datalist id="attention-backends">
            {#each ATTENTION_BACKENDS as backend (backend)}<option
                value={backend}
              ></option>{/each}
          </datalist>

          <label for={'pw-' + index}>prompt weighting</label>
          <input
            id={'pw-' + index}
            type="checkbox"
            class="check"
            checked={!!configuration.prompt_weighting}
            onchange={(e) => {
              if (e.currentTarget.checked) configuration.prompt_weighting = true
              else delete configuration.prompt_weighting
            }}
          />
        </div>
      </details>
    {:else if kind === 'task'}
      <div class="grid">
        <label for={'task-' + index}>command</label>
        <input
          id={'task-' + index}
          list="task-commands"
          bind:value={step.task.command}
          placeholder="e.g. upscale"
        />
      </div>
      <h3>arguments</h3>
      <ArgumentsEditor
        bind:args={step.task.arguments}
        componentType={step.task.command}
        target="task"
        listId={referenceListId}
      />
    {:else if kind === 'workflow'}
      <div class="grid">
        <label for={'wfpath-' + index}>path</label>
        <input
          id={'wfpath-' + index}
          list="workflow-files"
          bind:value={step.workflow.path}
          placeholder="Other.json, flux/FluxDev.json or builtin:augment_prompt.json"
        />
      </div>
      <h3>arguments</h3>
      <MappingEditor
        bind:args={step.workflow.arguments}
        suggestions={workflowVariables}
        listId={referenceListId}
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
  .step {
    margin-bottom: 0.7rem;
  }
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
  .flex {
    flex: 1;
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
    .grid,
    .grid2 {
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
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    margin: 1rem 0 0.5rem;
  }
  details {
    margin-top: 0.9rem;
    border-top: 1px solid var(--line);
    padding-top: 0.6rem;
  }
  summary {
    cursor: pointer;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    font-weight: 600;
    user-select: none;
  }
  summary:hover {
    color: var(--ink);
  }
  details[open] > summary {
    color: var(--accent);
  }
  summary :global(svg) {
    vertical-align: -2px;
    margin-right: 2px;
  }
  .section {
    margin-top: 0.7rem;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }
  .addrow {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    max-width: 300px;
  }
  .grid2 {
    display: grid;
    grid-template-columns: 150px minmax(0, 1fr);
    gap: 0.5rem 0.7rem;
    align-items: center;
  }
  .grid2 > label {
    font-weight: 600;
    color: var(--muted);
  }
  .check {
    width: auto;
    justify-self: start;
  }
  .inline-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }
  .inline-row select {
    max-width: 150px;
  }
  .num {
    max-width: 110px;
  }
  .error {
    color: var(--bad);
    font-size: 0.8rem;
  }
  .raw {
    margin-top: 0.8rem;
  }
  .raw textarea {
    font-family: ui-monospace, monospace;
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
  .digest {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    margin-top: var(--space-2);
  }
  .digestline {
    display: flex;
    gap: var(--space-2);
    align-items: baseline;
    background: transparent;
    border: 0;
    color: var(--ink);
    font-weight: 400;
    text-align: left;
    padding: 0.2rem 0.3rem;
    border-radius: var(--radius-1);
    font-size: 0.85rem;
  }
  .digestline:hover {
    background: var(--panel-2);
    filter: none;
  }
  .digestsection {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    flex: none;
    width: 90px;
  }
  .digesttext {
    overflow-wrap: anywhere;
  }
  .activebtn {
    border-color: var(--accent);
    color: var(--accent);
    position: relative;
    z-index: 1;
  }
</style>
