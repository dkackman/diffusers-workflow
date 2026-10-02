<script lang="ts">
  import Suggest from '../ui/Suggest.svelte'
  import { editorLists } from '../editorLists.svelte'
  import { Boxes, Layers, Timer, Zap } from '@lucide/svelte'
  import ArgumentsEditor from './ArgumentsEditor.svelte'
  import ComponentEditor from './ComponentEditor.svelte'
  import LorasEditor from './LorasEditor.svelte'
  import {
    ATTENTION_BACKENDS,
    CACHE_TYPES,
    COMPONENT_SLOTS,
    classDescription,
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

  const activeSlots = $derived(
    COMPONENT_SLOTS.filter((slot) => pipeline && slot in pipeline),
  )
  let addSlot = $state('')

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
    const schedulerType = pipeline?.scheduler?.configuration?.scheduler_type
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

<details open={activeSlots.length > 0 || openSection === 'components'}>
  <summary
    ><Boxes size={13} /> components
    <span class="muted">({activeSlots.length})</span></summary
  >
  <div class="section">
    {#each activeSlots as slot (slot)}
      <ComponentEditor
        {slot}
        bind:component={pipeline[slot]}
        suggestions={references}
        onremove={() => delete pipeline[slot]}
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
        title="declare the selected component on this pipeline">add</button
      >
    </div>
  </div>
</details>

<details open={(pipeline.loras ?? []).length > 0 || openSection === 'loras'}>
  <summary
    ><Layers size={13} /> LoRAs
    <span class="muted">({(pipeline.loras ?? []).length})</span></summary
  >
  <div class="section">
    <LorasEditor bind:pipeline />
  </div>
</details>

<details open={!!pipeline.scheduler || openSection === 'scheduler'}>
  <summary><Timer size={13} /> scheduler</summary>
  <div class="section grid2">
    <label for={'sched-' + index}>replace scheduler</label>
    <input
      id={'sched-' + index}
      type="checkbox"
      class="check"
      checked={!!pipeline.scheduler}
      onchange={(e) => toggleScheduler(e.currentTarget.checked)}
    />
    {#if pipeline.scheduler}
      <label for={'schedtype-' + index}>scheduler_type</label>
      <div>
        <Suggest
          id={'schedtype-' + index}
          suggestions={editorLists.schedulerClasses}
          bind:value={pipeline.scheduler.configuration.scheduler_type}
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
        bind:args={pipeline.scheduler.from_config_args}
        componentType={pipeline.scheduler.configuration.scheduler_type ?? ''}
        target="init"
        suggestions={references}
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
            setNumber(configuration.cache, 'threshold', e.currentTarget.value)}
        />
      {/if}
    </div>

    <label for={'attn-' + index}>attention backend</label>
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
  @container (max-width: 400px) {
    .grid2 {
      grid-template-columns: minmax(0, 1fr);
      gap: 0.2rem;
    }
  }
  .hint {
    font-size: 0.75rem;
    margin-top: 0.3rem;
  }
</style>
