<script lang="ts">
  import { Plus, Trash2 } from 'lucide-svelte'
  import { classDescription, widgetFor, coerce, isReference } from '../editor'
  import type { PipelineDescription } from '../types'

  let {
    args = $bindable(),
    componentType,
    target = 'call',
    hide = [],
  }: {
    args: Record<string, unknown>
    componentType: string
    target?: 'call' | 'init' | 'load'
    hide?: string[]
  } = $props()

  let description = $state<PipelineDescription | null>(null)
  let adding = $state('')

  $effect(() => {
    description = null
    if (componentType) {
      classDescription(componentType, target).then((d) => (description = d))
    }
  })

  const parameters = $derived(
    new Map((description?.parameters ?? []).map((p) => [p.name, p])),
  )
  const unused = $derived(
    (description?.parameters ?? []).filter(
      (p) => !(p.name in args) && !hide.includes(p.name),
    ),
  )
  const shownKeys = $derived(Object.keys(args).filter((key) => !hide.includes(key)))

  function displayValue(value: unknown): string {
    if (value === null || value === undefined) return ''
    if (typeof value === 'object') return JSON.stringify(value, null, 2)
    return String(value)
  }

  function update(key: string, raw: string) {
    args[key] = coerce(widgetFor(parameters.get(key), args[key]), raw)
  }

  function add() {
    if (!adding) return
    const parameter = parameters.get(adding)
    args[adding] = parameter?.default ?? ''
    adding = ''
  }

  function remove(key: string) {
    delete args[key]
  }
</script>

<div class="args">
  {#each shownKeys as key (key)}
    {@const parameter = parameters.get(key)}
    {@const widget = widgetFor(parameter, args[key])}
    <div class="row">
      <label for={'arg-' + key} title={parameter?.description ?? ''}>
        {key}{#if parameter?.required}<span class="req">*</span>{/if}
      </label>
      <div class="field">
        {#if widget === 'boolean' && !isReference(args[key])}
          <select
            id={'arg-' + key}
            value={String(args[key])}
            onchange={(e) => update(key, e.currentTarget.value)}
          >
            <option value="true">true</option>
            <option value="false">false</option>
          </select>
        {:else if widget === 'textarea' || widget === 'json'}
          <textarea
            id={'arg-' + key}
            rows="3"
            value={displayValue(args[key])}
            onchange={(e) => update(key, e.currentTarget.value)}
          ></textarea>
        {:else}
          <input
            id={'arg-' + key}
            value={displayValue(args[key])}
            onchange={(e) => update(key, e.currentTarget.value)}
          />
        {/if}
        {#if parameter?.description}
          <div class="hint muted">{parameter.description}</div>
        {/if}
      </div>
      <button class="quiet icon" title="remove" onclick={() => remove(key)}>
        <Trash2 size={14} />
      </button>
    </div>
  {/each}

  {#if unused.length}
    <div class="row add">
      <select bind:value={adding}>
        <option value="">add argument…</option>
        {#each unused as parameter}
          <option value={parameter.name} title={parameter.description ?? ''}>
            {parameter.name}{parameter.required ? ' *' : ''}
          </option>
        {/each}
      </select>
      <button class="quiet icon" onclick={add} disabled={!adding}>
        <Plus size={14} />
      </button>
    </div>
  {:else if componentType && !description}
    <div class="muted hint">no argument schema available for {componentType}</div>
  {/if}
</div>

<style>
  .args { display: flex; flex-direction: column; gap: 0.5rem; }
  .row { display: grid; grid-template-columns: 170px 1fr auto; gap: 0.7rem; align-items: start; }
  .row.add { grid-template-columns: 1fr auto; max-width: 340px; }
  label { padding-top: 0.42rem; font-weight: 600; color: var(--muted); overflow-wrap: anywhere; }
  .req { color: var(--warn); margin-left: 2px; }
  .hint { font-size: 0.75rem; margin-top: 0.15rem; max-width: 60ch; }
  .icon { display: inline-flex; align-items: center; padding: 0.4rem 0.5rem; }
</style>
