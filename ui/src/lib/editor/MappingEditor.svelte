<script lang="ts">
  import { Plus, Trash2 } from '@lucide/svelte'
  import { coerce, displayValue, isReference, widgetFor } from '../editor'
  import { promptLibrary } from '../promptlib.svelte'
  import { promptTooltip } from '../prompts'

  let {
    args = $bindable(),
    suggestions = [],
    listId = undefined,
  }: {
    args: Record<string, unknown>
    suggestions?: Array<{ name: string; hint?: string }>
    listId?: string
  } = $props()

  const uid = $props.id()
  let newKey = $state('')

  const unusedSuggestions = $derived(
    suggestions.filter((s) => !(s.name in args)),
  )
  const hintFor = (key: string) =>
    suggestions.find((s) => s.name === key)?.hint ?? ''

  function update(key: string, raw: string) {
    args[key] = coerce(widgetFor(undefined, args[key]), raw)
  }

  function add(key: string) {
    if (!key || key in args) return
    args[key] = ''
    newKey = ''
  }
</script>

<div class="mapping">
  {#each Object.keys(args) as key (key)}
    <label for={`${uid}-${key}`}>{key}</label>
    <input
      id={`${uid}-${key}`}
      class:ref={isReference(args[key])}
      list={listId}
      autocomplete="off"
      title={promptTooltip(args[key], promptLibrary.texts)}
      value={displayValue(args[key])}
      placeholder={hintFor(key)}
      onchange={(e) => update(key, e.currentTarget.value)}
    />
    <button
      class="quiet icon"
      onclick={() => delete args[key]}
      title="remove this argument"
      aria-label="remove this argument"
    >
      <Trash2 size={14} />
    </button>
  {/each}

  {#if unusedSuggestions.length}
    <select
      value=""
      title="add one of the target's arguments"
      onchange={(e) => {
        add(e.currentTarget.value)
        e.currentTarget.value = ''
      }}
    >
      <option value="">add argument…</option>
      {#each unusedSuggestions as suggestion (suggestion.name)}
        <option value={suggestion.name} title={suggestion.hint ?? ''}>
          {suggestion.name}
        </option>
      {/each}
    </select>
  {:else}
    <input placeholder="new argument name…" bind:value={newKey} />
  {/if}
  <button
    class="quiet icon"
    onclick={() => add(newKey)}
    disabled={unusedSuggestions.length > 0 || !newKey}
    title="add this argument"
    aria-label="add this argument"
  >
    <Plus size={14} />
  </button>
  <span></span>
</div>

<style>
  .mapping {
    display: grid;
    grid-template-columns: minmax(140px, auto) 1fr auto;
    gap: 0.5rem 0.8rem;
    align-items: center;
  }
  .mapping label {
    font-weight: 600;
    color: var(--muted);
    overflow-wrap: anywhere;
  }
  .mapping select {
    grid-column: 1 / span 2;
    max-width: 300px;
  }
  .mapping input[placeholder='new argument name…'] {
    grid-column: 1 / span 2;
    max-width: 300px;
  }
  .icon {
    display: inline-flex;
    padding: 0.3rem 0.45rem;
    justify-self: start;
  }
</style>
