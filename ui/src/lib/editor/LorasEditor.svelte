<script lang="ts">
  import { setNumber } from '../editor'
  import { Plus, Trash2 } from '@lucide/svelte'

  let { pipeline = $bindable() }: { pipeline: Record<string, any> } = $props()

  const loras = $derived(pipeline.loras ?? [])

  function add() {
    pipeline.loras = [...(pipeline.loras ?? []), { model_name: '', scale: 1.0 }]
  }

  function remove(index: number) {
    pipeline.loras.splice(index, 1)
    if (pipeline.loras.length === 0) delete pipeline.loras
  }
</script>

<div class="loras">
  {#each loras as lora, index (index)}
    <div class="row">
      <input bind:value={lora.model_name} placeholder="org/lora-repo or path" />
      <input
        value={lora.weight_name ?? ''}
        placeholder="weight file (optional)"
        onchange={(e) => {
          const v = e.currentTarget.value
          if (v) lora.weight_name = v
          else delete lora.weight_name
        }}
      />
      <input
        class="scale"
        value={lora.scale ?? ''}
        placeholder="scale"
        onchange={(e) => setNumber(lora, 'scale', e.currentTarget.value)}
      />
      <button class="quiet icon" onclick={() => remove(index)} title="remove">
        <Trash2 size={14} />
      </button>
    </div>
  {/each}
  <button
    class="quiet withicon"
    onclick={add}
    title="add a LoRA adapter to load onto this pipeline"
  >
    <Plus size={14} />add LoRA
  </button>
</div>

<style>
  .loras {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    align-items: flex-start;
  }
  .row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) 80px auto;
    gap: 0.5rem;
    width: 100%;
    align-items: center;
  }
  /* Two repo fields side by side need ~460px; below that they stack and the
     scale keeps the remove button company on the last line */
  @container (max-width: 460px) {
    .row {
      grid-template-columns: minmax(0, 1fr) auto;
    }
    .row > input:not(.scale) {
      grid-column: 1 / -1;
    }
  }
  .icon {
    display: inline-flex;
    padding: 0.3rem 0.45rem;
  }
  .withicon {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
</style>
