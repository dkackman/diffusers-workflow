<script lang="ts">
  import { Plus, Trash2 } from '@lucide/svelte'
  import {
    classDescription,
    coerce,
    displayValue,
    isReference,
    widgetFor,
  } from '../editor'
  import { promptLibrary } from '../promptlib.svelte'
  import { promptTooltip } from '../prompts'
  import type { PipelineDescription } from '../types'

  let {
    args = $bindable(),
    componentType,
    target = 'call',
    hide = [],
    listId = undefined,
  }: {
    args: Record<string, unknown>
    componentType: string
    target?: 'call' | 'init' | 'load' | 'task'
    hide?: string[]
    listId?: string
  } = $props()

  let description = $state<PipelineDescription | null>(null)
  let adding = $state('')

  // Debounced: the class name arrives keystroke by keystroke, and every
  // prefix would otherwise fire (and cache) a doomed lookup
  $effect(() => {
    description = null
    if (!componentType) return
    const wanted = componentType
    const timer = setTimeout(() => {
      classDescription(wanted, target).then((d) => {
        if (wanted === componentType) description = d
      })
    }, 300)
    return () => clearTimeout(timer)
  })

  const parameters = $derived(
    new Map((description?.parameters ?? []).map((p) => [p.name, p])),
  )
  const unused = $derived(
    (description?.parameters ?? []).filter(
      (p) => !(p.name in args) && !hide.includes(p.name),
    ),
  )
  const shownKeys = $derived(
    Object.keys(args).filter((key) => !hide.includes(key)),
  )

  function update(key: string, raw: string) {
    args[key] = coerce(widgetFor(parameters.get(key), args[key]), raw)
  }

  function add() {
    if (!adding) return
    const parameter = parameters.get(adding)
    args[adding] = parameter?.default ?? ''
    adding = ''
  }

  // Free-form additions belong wherever the schema is open-ended: the
  // callable takes **kwargs, or no schema resolved at all
  const openEnded = $derived(!description || description.accepts_kwargs)
  let customName = $state('')

  function addCustom() {
    const name = customName.trim()
    if (!name || name in args) return
    args[name] = parameters.get(name)?.default ?? ''
    customName = ''
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
          <!-- A reference is never a textarea: widgetFor routes it to the
               input branch below, which carries the datalist and tooltip -->
          <textarea
            id={'arg-' + key}
            spellcheck={widget === 'textarea'}
            rows="3"
            value={displayValue(args[key], true)}
            onchange={(e) => update(key, e.currentTarget.value)}></textarea>
        {:else}
          <input
            id={'arg-' + key}
            class:ref={isReference(args[key])}
            list={listId}
            autocomplete="off"
            title={promptTooltip(args[key], promptLibrary.texts)}
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
        {#each unused as parameter (parameter.name)}
          <option value={parameter.name} title={parameter.description ?? ''}>
            {parameter.name}{parameter.required ? ' *' : ''}
          </option>
        {/each}
      </select>
      <button
        class="quiet icon"
        onclick={add}
        disabled={!adding}
        title="add the selected argument"
        aria-label="add the selected argument"
      >
        <Plus size={14} />
      </button>
    </div>
  {/if}
  {#if openEnded && componentType}
    <div class="row add">
      <input
        placeholder="add custom argument…"
        bind:value={customName}
        onkeydown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault()
            addCustom()
          }
        }}
        title={description
          ? 'this command also accepts arguments beyond the named ones'
          : 'no argument schema available - arguments are free-form'}
      />
      <button
        class="quiet icon"
        onclick={addCustom}
        disabled={!customName.trim()}
        title="add this argument"
        aria-label="add this argument"
      >
        <Plus size={14} />
      </button>
    </div>
  {/if}
</div>

<style>
  .args {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .row {
    display: grid;
    grid-template-columns: 170px 1fr auto;
    gap: 0.7rem;
    align-items: start;
  }
  .row.add {
    grid-template-columns: 1fr auto;
    max-width: 340px;
  }
  label {
    padding-top: 0.42rem;
    font-weight: 600;
    color: var(--muted);
    overflow-wrap: anywhere;
  }
  .req {
    color: var(--warn);
    margin-left: 2px;
  }
  .hint {
    font-size: 0.75rem;
    margin-top: 0.15rem;
    max-width: 60ch;
  }
  .icon {
    display: inline-flex;
    align-items: center;
    padding: 0.4rem 0.5rem;
  }
</style>
