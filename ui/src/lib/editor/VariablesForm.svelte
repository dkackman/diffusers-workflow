<script lang="ts">
  import { Plus } from '@lucide/svelte'
  import ExpandingText from './ExpandingText.svelte'
  import { coerce, displayValue, isReference, widgetFor } from '../editor'
  import { promptLibrary } from '../promptlib.svelte'
  import { promptListId, promptTooltip } from '../prompts'

  let {
    mode,
    variables = $bindable(),
    overrides = $bindable({}),
    idPrefix,
  }: {
    /** define: edit the workflow's own defaults (add/remove/coerce).
     * override: fill per-run values over read-only defaults. */
    mode: 'define' | 'override'
    variables: Record<string, unknown>
    overrides?: Record<string, string>
    idPrefix: string
  } = $props()

  const keys = $derived(Object.keys(variables ?? {}))

  // Raw text as typed, per variable - the committed value only updates on
  // blur, and the prompt datalist should attach as soon as prompt: is typed
  let drafts = $state<Record<string, string>>({})
  let newVariable = $state('')

  function shown(key: string): string {
    if (mode === 'override') return overrides[key] ?? ''
    return displayValue(variables[key])
  }

  /** The value the field is effectively holding - an override falls back
   * to the default it would leave in place. */
  function effective(key: string): unknown {
    if (mode === 'override') return overrides[key] || variables[key]
    return variables[key]
  }

  function commit(key: string, raw: string) {
    if (mode === 'override') {
      overrides[key] = raw
      return
    }
    // Coerce to the default's type - schema validation runs on the
    // definition itself, so "9" where 9 belongs breaks the workflow
    variables[key] = coerce(widgetFor(undefined, variables[key]), raw)
  }

  function draft(key: string, raw: string) {
    if (mode === 'override') overrides[key] = raw
    else drafts[key] = raw
  }

  function addVariable() {
    if (!newVariable) return
    variables = { ...(variables ?? {}), [newVariable]: '' }
    newVariable = ''
  }

  function removeVariable(key: string) {
    delete variables[key]
  }
</script>

<div class="vars" class:define={mode === 'define'}>
  {#each keys as key (key)}
    <label for={idPrefix + key}>{key}</label>
    <ExpandingText
      id={idPrefix + key}
      value={shown(key)}
      placeholder={mode === 'override' ? displayValue(variables[key]) : ''}
      refStyle={isReference(effective(key))}
      listId={promptListId(
        mode === 'override' ? effective(key) : (drafts[key] ?? variables[key]),
      )}
      title={promptTooltip(effective(key), promptLibrary.texts)}
      oninput={(raw) => draft(key, raw)}
      onchange={(raw) => commit(key, raw)}
      onpromptpick={(name) => commit(key, 'prompt:' + name)}
    />
    {#if mode === 'define'}
      <button
        class="quiet icon"
        onclick={() => removeVariable(key)}
        title="remove this variable"
        aria-label="remove this variable"
      >
        ×
      </button>
    {/if}
  {/each}
  {#if mode === 'define'}
    <input placeholder="new variable name…" bind:value={newVariable} />
    <button
      class="quiet withicon addvar"
      onclick={addVariable}
      disabled={!newVariable}
      title="add this variable to the workflow"
    >
      <Plus size={14} />add
    </button>
    <span></span>
  {/if}
</div>

<style>
  .vars {
    display: grid;
    grid-template-columns: minmax(140px, 40%) minmax(0, 1fr);
    gap: var(--space-2) var(--space-3);
    align-items: start;
  }
  .vars.define {
    grid-template-columns: minmax(140px, 40%) minmax(0, 1fr) auto;
  }
  @container (max-width: 400px) {
    .vars,
    .vars.define {
      grid-template-columns: minmax(0, 1fr) auto;
      gap: var(--space-1) var(--space-2);
    }
    .vars > label {
      grid-column: 1 / -1;
      padding-top: 0;
    }
  }
  .vars label {
    padding-top: 0.45rem;
    font-weight: 600;
    color: var(--muted);
    overflow-wrap: anywhere;
  }
  .icon {
    padding: 0.3rem 0.55rem;
  }
  .withicon {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  .addvar {
    justify-self: start;
  }
</style>
