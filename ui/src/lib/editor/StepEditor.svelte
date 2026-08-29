<script lang="ts">
  import { ChevronDown, ChevronRight, ChevronUp, Trash2 } from 'lucide-svelte'
  import ArgumentsEditor from './ArgumentsEditor.svelte'
  import { TORCH_DTYPES, CONTENT_TYPES } from '../editor'

  let {
    step = $bindable(),
    index,
    count,
    pipelines,
    onremove,
    onmove,
  }: {
    step: Record<string, any>
    index: number
    count: number
    pipelines: string[]
    onremove: () => void
    onmove: (delta: number) => void
  } = $props()

  let open = $state(true)

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

  // Non-pipeline steps are edited as raw JSON for now - everything the
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
    if (kind !== 'pipeline') openRaw()
  })

  const configuration = $derived(step.pipeline?.configuration ?? {})
  const pretrained = $derived(step.pipeline?.from_pretrained_arguments ?? {})
</script>

<div class="panel step">
  <div class="bar">
    <button class="quiet icon" onclick={() => (open = !open)}>
      {#if open}<ChevronDown size={15} />{:else}<ChevronRight size={15} />{/if}
    </button>
    <input class="name" bind:value={step.name} />
    <span class="kind muted">{kind}</span>
    <span class="flex"></span>
    <button class="quiet icon" disabled={index === 0} onclick={() => onmove(-1)} title="move up">
      <ChevronUp size={15} />
    </button>
    <button class="quiet icon" disabled={index === count - 1} onclick={() => onmove(1)} title="move down">
      <ChevronDown size={15} />
    </button>
    <button class="quiet icon" onclick={onremove} title="remove step">
      <Trash2 size={15} />
    </button>
  </div>

  {#if open}
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
          {#each TORCH_DTYPES as dtype}<option>{dtype}</option>{/each}
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
          {#each CONTENT_TYPES as contentType}<option>{contentType}</option>{/each}
        </select>
      </div>

      <h3>arguments</h3>
      <ArgumentsEditor
        bind:args={step.pipeline.arguments}
        componentType={configuration.component_type ?? ''}
      />
    {:else}
      <div class="raw">
        <textarea rows="8" bind:value={rawDraft} onchange={applyRaw}></textarea>
        {#if rawError}<div class="error">{rawError}</div>{/if}
        <div class="muted hint">
          {kind} steps are edited as JSON for now - changes apply on blur
        </div>
      </div>
    {/if}
  {/if}
</div>

<style>
  .step { margin-bottom: 0.7rem; }
  .bar { display: flex; align-items: center; gap: 0.5rem; }
  .name { max-width: 220px; font-weight: 600; }
  .kind { font-size: 0.75rem; }
  .flex { flex: 1; }
  .icon { display: inline-flex; align-items: center; padding: 0.3rem 0.45rem; }
  .grid {
    display: grid; grid-template-columns: 170px minmax(200px, 480px);
    gap: 0.5rem 0.7rem; margin-top: 0.8rem; align-items: center;
  }
  .grid label { font-weight: 600; color: var(--muted); }
  h3 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;
       color: var(--muted); margin: 1rem 0 0.5rem; }
  .raw { margin-top: 0.8rem; }
  .raw textarea { font-family: ui-monospace, monospace; font-size: 0.82rem; }
  .error { color: var(--bad); font-size: 0.8rem; margin-top: 0.3rem; }
  .hint { font-size: 0.75rem; margin-top: 0.3rem; }
</style>
