<script lang="ts">
  import { Trash2 } from '@lucide/svelte'
  import ArgumentsEditor from './ArgumentsEditor.svelte'
  import QuantizationEditor from './QuantizationEditor.svelte'

  let {
    slot,
    component = $bindable(),
    listId = undefined,
    onremove,
  }: {
    slot: string
    component: Record<string, any>
    listId?: string
    onremove: () => void
  } = $props()

  // The schema requires these blocks; older hand-written files may omit one.
  // Initialized in setup (runs once) rather than an effect - an effect would
  // subscribe itself to the properties it writes
  component.configuration ??= { component_type: '' }
  component.from_pretrained_arguments ??= {}

  const groupOffload = $derived(component.group_offload ?? null)

  function toggleGroupOffload(enabled: boolean) {
    if (enabled) {
      component.group_offload = component.group_offload ?? {
        offload_type: 'leaf_level',
        use_stream: true,
      }
    } else {
      delete component.group_offload
    }
  }
</script>

<div class="component">
  <div class="bar">
    <strong>{slot}</strong>
    <span class="flex"></span>
    <button class="quiet icon" onclick={onremove} title="remove component">
      <Trash2 size={14} />
    </button>
  </div>

  {#if component.configuration}
    <div class="grid">
      <label for={slot + '-type'}>class</label>
      <input
        id={slot + '-type'}
        list="model-classes"
        bind:value={component.configuration.component_type}
        placeholder="e.g. FluxTransformer2DModel"
      />

      <label for={slot + '-model'}>model</label>
      <input
        id={slot + '-model'}
        bind:value={component.from_pretrained_arguments.model_name}
        placeholder="org/model or local path"
      />

      <label for={slot + '-load'}>load args</label>
      <ArgumentsEditor
        bind:args={component.from_pretrained_arguments}
        componentType={component.configuration.component_type ?? ''}
        target="load"
        hide={['model_name']}
        {listId}
      />

      <label for={slot + '-device'}>device</label>
      <input
        id={slot + '-device'}
        value={component.configuration.device ?? ''}
        placeholder="pipeline default"
        onchange={(e) => {
          const v = e.currentTarget.value
          if (v) component.configuration.device = v
          else delete component.configuration.device
        }}
      />

      <label for={slot + '-quant'}>quantization</label>
      <QuantizationEditor bind:component {listId} />

      <label for={slot + '-go'}>group offload</label>
      <div class="gofield">
        <input
          id={slot + '-go'}
          type="checkbox"
          checked={groupOffload !== null}
          onchange={(e) => toggleGroupOffload(e.currentTarget.checked)}
        />
        {#if groupOffload}
          <select bind:value={groupOffload.offload_type}>
            <option value="leaf_level">leaf_level</option>
            <option value="block_level">block_level</option>
          </select>
          <label class="inline">
            <input type="checkbox" bind:checked={groupOffload.use_stream} /> use_stream
          </label>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .component {
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 0.7rem 0.8rem;
    background: var(--panel-2);
  }
  .bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
  }
  .flex {
    flex: 1;
  }
  .icon {
    display: inline-flex;
    padding: 0.25rem 0.4rem;
  }
  .grid {
    display: grid;
    grid-template-columns: 120px 1fr;
    gap: 0.45rem 0.7rem;
    align-items: start;
  }
  .grid > label {
    font-weight: 600;
    color: var(--muted);
    padding-top: 0.35rem;
  }
  .gofield {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
  }
  .gofield input[type='checkbox'] {
    width: auto;
  }
  .gofield select {
    max-width: 150px;
  }
  .inline {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    color: var(--muted);
  }
  .inline input {
    width: auto;
  }
</style>
