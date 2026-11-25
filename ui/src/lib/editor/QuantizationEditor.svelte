<script lang="ts">
  import Suggest from '../ui/Suggest.svelte'
  import { editorLists } from '../editorLists.svelte'
  import ArgumentsEditor from './ArgumentsEditor.svelte'
  import { QUANT_PRESETS } from '../editor'
  import type { PipelineComponent } from '../types'

  let {
    component = $bindable(),
    suggestions = [],
  }: { component: PipelineComponent; suggestions?: readonly string[] } =
    $props()

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
    {#each Object.keys(QUANT_PRESETS) as name (name)}<option value={name}>
        {name}
      </option>{/each}
    {#if presetName === 'custom'}<option value="custom">
        custom ({configType})
      </option>{/if}
  </select>

  {#if current}
    {@const quant = current}
    <Suggest
      class="ctype"
      suggestions={editorLists.quantizationClasses}
      bind:value={
        () => quant.configuration?.config_type ?? '',
        (v) => {
          quant.configuration ??= {}
          quant.configuration.config_type = v
        }
      }
      title="config_type - any importable quantization config class"
    />
    <ArgumentsEditor
      bind:args={quant.arguments}
      componentType={quant.configuration?.config_type ?? ''}
      target="init"
      {suggestions}
    />
  {/if}
</div>

<style>
  .quant {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  :global(input.ctype) {
    font-family: var(--font-mono);
    font-size: 0.8rem;
  }
</style>
