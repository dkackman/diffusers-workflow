<script lang="ts">
  import ArgumentsEditor from './ArgumentsEditor.svelte'
  import { QUANT_PRESETS } from '../editor'

  let {
    component = $bindable(),
    listId = undefined,
  }: { component: Record<string, any>; listId?: string } = $props()

  const current = $derived(component.quantization_config ?? null)
  const configType = $derived(current?.configuration?.config_type ?? '')

  // Which preset matches the current config_type (for the select's value)
  const presetName = $derived.by(() => {
    if (!current) return 'none'
    for (const [name, preset] of Object.entries(QUANT_PRESETS)) {
      if (preset && preset.config_type === configType) return name
    }
    return 'custom'
  })

  function applyPreset(name: string) {
    const preset = QUANT_PRESETS[name]
    if (name === 'none') {
      delete component.quantization_config
    } else if (preset) {
      component.quantization_config = {
        configuration: { config_type: preset.config_type },
        arguments: { ...preset.arguments },
      }
    }
  }

</script>

<div class="quant">
  <select
    value={presetName}
    onchange={(e) => applyPreset(e.currentTarget.value)}
    title="quantization"
  >
    {#each Object.keys(QUANT_PRESETS) as name}<option value={name}>{name}</option>{/each}
    {#if presetName === 'custom'}<option value="custom">custom ({configType})</option>{/if}
  </select>

  {#if current}
    <input
      class="ctype"
      list="quantization-classes"
      bind:value={current.configuration.config_type}
      title="config_type - any importable quantization config class"
    />
    <ArgumentsEditor
      bind:args={current.arguments}
      componentType={current.configuration.config_type ?? ''}
      target="init"
      listId={listId}
    />
  {/if}
</div>

<style>
  .quant { display: flex; flex-direction: column; gap: 0.4rem; }
  .ctype { font-family: ui-monospace, monospace; font-size: 0.8rem; }
</style>
