<script lang="ts">
  import Suggest from '../ui/Suggest.svelte'
  import { editorLists } from '../editorLists.svelte'
  import { attentionBackends } from '../attentionBackends.svelte'
  import { Boxes, Layers, Timer, Zap } from '@lucide/svelte'
  import ArgumentsEditor from './ArgumentsEditor.svelte'
  import ComponentEditor from './ComponentEditor.svelte'
  import LorasEditor from './LorasEditor.svelte'
  import { latestOnly } from '../latest'
  import { sleep } from '../poll'
  import { resource } from '../resource.svelte'
  import { slotsFrom } from './componentSlots'
  import {
    ATTENTION_BACKENDS,
    CACHE_TYPES,
    COMPONENT_SLOTS,
    OPTION_KEYS,
    classDescription,
    isReference,
    emptyComponent,
    setNumber,
  } from '../editor'

  // A pipeline step's optional blocks, each a <details> that opens on its
  // own when the step sets it or a compact digest line jumped to it
  let {
    pipeline = $bindable(),
    index,
    openSection,
    references,
  }: {
    pipeline: Record<string, any>
    index: number
    openSection: string
    references: string[]
  } = $props()

  const configuration = $derived(pipeline.configuration ?? {})

  // ---- optional blocks: components, loras, scheduler, acceleration ----

  // Any key outside COMPONENT_SLOTS and OPTION_KEYS whose value carries
  // from_pretrained_arguments is a component the engine loads under that name
  const activeSlots = $derived.by(() => {
    if (!pipeline) return []
    const known: readonly string[] = COMPONENT_SLOTS
    const extra = Object.keys(pipeline).filter(
      (key) =>
        !known.includes(key) &&
        !(OPTION_KEYS as readonly string[]).includes(key) &&
        pipeline[key] !== null &&
        typeof pipeline[key] === 'object' &&
        'from_pretrained_arguments' in pipeline[key],
    )
    return [...known.filter((slot) => slot in pipeline), ...extra]
  })
  let addSlot = $state('')

  // The pipeline class's own constructor names its components; a modular
  // pipeline or an unknown class keeps the schema's list. Debounced like
  // ArgumentsEditor's: the class name arrives keystroke by keystroke
  const classTyping = latestOnly()
  const classLookup = resource(
    async () => {
      const wanted = pipeline?.configuration?.component_type
      const current = classTyping()
      if (!wanted) return null
      await sleep(300)
      return current() ? classDescription(wanted, 'init') : null
    },
    { watch: [] },
  )
  const pipelineSlots = $derived(
    slotsFrom(
      classLookup.status === 'ready' ? (classLookup.data ?? null) : null,
    ),
  )
  // The class's own list is on its way: say so rather than offer the
  // schema's, which the answer may then narrow
  const slotsLoading = $derived(
    !!pipeline?.configuration?.component_type &&
      classLookup.status === 'loading',
  )
  const offeredSlots = $derived(
    (pipelineSlots ?? [...COMPONENT_SLOTS]).filter(
      (slot) => !activeSlots.includes(slot),
    ),
  )

  function addComponent() {
    if (!addSlot) return
    pipeline[addSlot] = emptyComponent()
    addSlot = ''
  }

  function toggleScheduler(enabled: boolean) {
    if (enabled) {
      pipeline.scheduler = pipeline.scheduler ?? {
        configuration: { scheduler_type: '' },
        from_config_args: {},
      }
    } else {
      delete pipeline.scheduler
    }
  }

  // A scheduler block is a replacement class, a shift on the pipeline's own
  // scheduler, or both (the schema takes either) - H3 sets only a shift
  type SchedulerKey = 'scheduler' | 'audio_scheduler'

  function setSchedulerType(key: SchedulerKey, value: string) {
    const scheduler = pipeline[key]
    if (!value) {
      delete scheduler.configuration
    } else if (scheduler.configuration) {
      scheduler.configuration.scheduler_type = value
    } else {
      scheduler.configuration = { scheduler_type: value }
      scheduler.from_config_args = scheduler.from_config_args ?? {}
    }
  }

  function setShift(key: SchedulerKey, raw: string) {
    const value = raw.trim()
    const parsed = Number(value)
    if (!value) delete pipeline[key].shift
    else pipeline[key].shift = Number.isNaN(parsed) ? value : parsed
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

  // Only complete-looking names - prefixes typed on the way to a real one
  // would each fire a doomed lookup. A scheduler's compatibles never change
  // while the server runs, so the lookup watches nothing
  const typing = latestOnly()
  const schedulerLookup = resource(
    async () => {
      const schedulerType = pipeline?.scheduler?.configuration?.scheduler_type
      const current = typing()
      if (!schedulerType || !schedulerType.endsWith('Scheduler')) return []
      await sleep(300)
      if (!current()) return []
      return (await classDescription(schedulerType, 'init'))?.compatibles ?? []
    },
    { watch: [] },
  )
  const compatibles = $derived(
    schedulerLookup.status === 'ready' ? (schedulerLookup.data ?? []) : [],
  )
</script>

<details open={activeSlots.length > 0 || openSection === 'components'}>
  <summary>
    <Boxes size={13} /> components
    <span class="muted">({activeSlots.length})</span>
  </summary>
  <div class="section">
    {#each activeSlots as slot (slot)}
      {#if pipelineSlots && !pipelineSlots.includes(slot)}
        <span class="warn">
          not a component of {configuration.component_type}
        </span>
      {/if}
      <ComponentEditor
        {slot}
        bind:component={pipeline[slot]}
        suggestions={references}
        onremove={() => delete pipeline[slot]}
      />
    {/each}
    <div class="addrow">
      <!-- The loading label is a block of its own: an attribute or a text
           bound to the lookup would share the effect that sets each fold's
           `open`, and its answer would shut a fold the person just opened -->
      <select bind:value={addSlot} aria-label="component to add">
        {#if slotsLoading}
          <option value="">reading the pipeline…</option>
        {:else}
          <option value="">add component…</option>
        {/if}
        {#each offeredSlots as slot (slot)}
          <option value={slot}>{slot}</option>
        {/each}
      </select>
      <button
        class="quiet"
        onclick={addComponent}
        disabled={!addSlot}
        title="declare the selected component on this pipeline"
      >
        add
      </button>
    </div>
  </div>
</details>

<details open={(pipeline.loras ?? []).length > 0 || openSection === 'loras'}>
  <summary>
    <Layers size={13} /> LoRAs
    <span class="muted">({(pipeline.loras ?? []).length})</span>
  </summary>
  <div class="section">
    <LorasEditor bind:pipeline />
  </div>
</details>

{#snippet schedulerFields(key: SchedulerKey)}
  {@const scheduler = pipeline[key]}
  <label for={`${key}type-${index}`}>scheduler_type</label>
  <div>
    <Suggest
      id={`${key}type-${index}`}
      suggestions={editorLists.schedulerClasses}
      bind:value={
        () => scheduler.configuration?.scheduler_type ?? '',
        (v) => setSchedulerType(key, v)
      }
      placeholder="the pipeline's own - e.g. FlowMatchEulerDiscreteScheduler"
    />
    {#if key === 'scheduler' && compatibles.length}
      <div class="muted hint">
        interchangeable with: {compatibles
          .slice(0, 6)
          .join(', ')}{compatibles.length > 6 ? ', …' : ''}
      </div>
    {/if}
  </div>
  <label for={`${key}shift-${index}`}>shift</label>
  <input
    id={`${key}shift-${index}`}
    value={scheduler.shift ?? ''}
    onchange={(e) => setShift(key, e.currentTarget.value)}
    placeholder="number or variable:name"
  />
  {#if scheduler.configuration && scheduler.from_config_args}
    <label for={`${key}args-${index}`}>from_config_args</label>
    <ArgumentsEditor
      bind:args={scheduler.from_config_args}
      componentType={scheduler.configuration.scheduler_type ?? ''}
      target="init"
      suggestions={references}
    />
  {/if}
{/snippet}

<details
  open={!!pipeline.scheduler ||
    !!pipeline.audio_scheduler ||
    openSection === 'scheduler'}
>
  <summary><Timer size={13} /> scheduler</summary>
  <div class="section formgrid">
    <label for={'sched-' + index}>replace scheduler</label>
    <input
      id={'sched-' + index}
      type="checkbox"
      class="check"
      checked={!!pipeline.scheduler}
      onchange={(e) => toggleScheduler(e.currentTarget.checked)}
    />
    {#if pipeline.scheduler}
      {@render schedulerFields('scheduler')}
    {/if}
    {#if pipeline.audio_scheduler}
      <span class="muted subhead">audio scheduler</span>
      <span></span>
      {@render schedulerFields('audio_scheduler')}
    {/if}
  </div>
</details>

<details
  open={!!configuration.cache ||
    !!configuration.attention_backend ||
    openSection === 'acceleration'}
>
  <summary><Zap size={13} /> acceleration</summary>
  <div class="section formgrid">
    <label for={'cache-' + index}>cache</label>
    <div class="inline-row">
      <input
        id={'cache-' + index}
        type="checkbox"
        class="check"
        checked={!!configuration.cache}
        onchange={(e) => toggleCache(e.currentTarget.checked)}
      />
      {#if typeof configuration.cache === 'string'}
        <input
          placeholder="a reference, e.g. variable:cache"
          value={configuration.cache}
          onchange={(e) => {
            const v = e.currentTarget.value.trim()
            if (v) configuration.cache = v
            else delete configuration.cache
          }}
        />
      {:else if configuration.cache}
        <select bind:value={configuration.cache.type}>
          {#each CACHE_TYPES as cacheType (cacheType)}<option>
              {cacheType}
            </option>{/each}
        </select>
        <input
          class="num"
          placeholder="threshold"
          value={configuration.cache.threshold ?? ''}
          onchange={(e) =>
            setNumber(configuration.cache, 'threshold', e.currentTarget.value)}
        />
      {/if}
    </div>

    <label for={'attn-' + index}>attention backend</label>
    {#if attentionBackends.list.length && typeof (configuration.attention_backend ?? '') === 'string' && !isReference(configuration.attention_backend)}
      {@const current = configuration.attention_backend ?? ''}
      <select
        id={'attn-' + index}
        value={current}
        onchange={(e) => {
          if (e.currentTarget.value)
            configuration.attention_backend = e.currentTarget.value
          else delete configuration.attention_backend
        }}
      >
        <option value="">pipeline default</option>
        {#if current && !attentionBackends.list.includes(current)}
          <option value={current}>
            {current} (not available on this server)
          </option>
        {/if}
        {#each attentionBackends.list as backend (backend)}
          <option value={backend}>{backend}</option>
        {/each}
      </select>
    {:else}
      <Suggest
        id={'attn-' + index}
        suggestions={ATTENTION_BACKENDS}
        value={configuration.attention_backend ?? ''}
        placeholder="pipeline default"
        onchange={(v) => {
          if (v) configuration.attention_backend = v
          else delete configuration.attention_backend
        }}
      />
    {/if}

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

<style>
  details {
    margin-top: 0.9rem;
    border-top: 1px solid var(--line);
    padding-top: 0.6rem;
  }
  summary {
    cursor: pointer;
    font-size: 0.8rem;
    text-transform: none;
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
  .subhead {
    font-weight: 600;
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
  .hint {
    margin-top: 0.3rem;
  }
</style>
