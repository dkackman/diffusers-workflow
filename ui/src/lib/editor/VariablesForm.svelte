<script lang="ts">
  import { PROMPT, reference } from '../references'
  import { Plus, Trash2 } from '@lucide/svelte'
  import ExpandingText from './ExpandingText.svelte'
  import { coerce, displayValue, isReference, widgetFor } from '../editor'
  import { promptLibrary } from '../promptlib.svelte'
  import { promptTooltip } from '../prompts'
  import { referenceCompletions } from '../refcomplete.svelte'
  import BrowseButton from '../assets/BrowseButton.svelte'
  import { attentionBackends } from '../attentionBackends.svelte'

  let {
    mode,
    variables = $bindable(),
    overrides = $bindable({}),
    idPrefix,
    backendVariables = new Set<string>(),
  }: {
    /** define: edit the workflow's own defaults (add/remove/coerce).
     * override: fill per-run values over read-only defaults. */
    mode: 'define' | 'override'
    variables: Record<string, unknown>
    overrides?: Record<string, string>
    idPrefix: string
    /** Variables that feed an attention_backend: a dropdown of the
     * server's own list while it has one. */
    backendVariables?: Set<string>
  } = $props()

  const keys = $derived(Object.keys(variables ?? {}))

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

  /** A backend variable is a dropdown when the server listed backends and
   * both what it holds and what it would override are plain names - a
   * reference or a non-string stays the text field. */
  function isBackendSelect(key: string): boolean {
    if (!backendVariables.has(key) || !attentionBackends.list.length) {
      return false
    }
    const plain = (v: unknown) =>
      v === undefined || (typeof v === 'string' && !isReference(v))
    return plain(variables[key]) && (mode === 'define' || plain(overrides[key]))
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

  // An override is live as it is typed, so Run takes what is in the box
  // even before it blurs; a definition commits on change, coerced
  function draft(key: string, raw: string) {
    if (mode === 'override') overrides[key] = raw
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
    <div class="field">
      {#if isBackendSelect(key)}
        {@const current = shown(key)}
        <select
          id={idPrefix + key}
          value={current}
          onchange={(e) => commit(key, e.currentTarget.value)}
        >
          <option value="">
            {mode === 'override'
              ? `default (${displayValue(variables[key]) || 'none'})`
              : 'default (none)'}
          </option>
          {#if current && !attentionBackends.list.includes(current)}
            <option value={current}>
              {current} (not available on this server)
            </option>
          {/if}
          {#each attentionBackends.list as backend (backend)}
            <option value={backend}>{backend}</option>
          {/each}
        </select>
      {:else}
        <ExpandingText
          id={idPrefix + key}
          value={shown(key)}
          placeholder={mode === 'override' ? displayValue(variables[key]) : ''}
          refStyle={isReference(effective(key))}
          complete={(text) => referenceCompletions(text)}
          title={promptTooltip(effective(key), promptLibrary.texts)}
          oninput={(raw) => draft(key, raw)}
          onchange={(raw) => commit(key, raw)}
          onpromptpick={(name) => commit(key, reference(PROMPT, name))}
        />
      {/if}
      <BrowseButton
        value={effective(key)}
        onpick={(value) => commit(key, value)}
      />
    </div>
    {#if mode === 'define'}
      <button
        class="quiet icon"
        onclick={() => removeVariable(key)}
        title="remove this variable"
        aria-label="remove this variable"
      >
        <Trash2 size={14} />
      </button>
    {/if}
  {/each}
  {#if mode === 'define'}
    <input placeholder="new variable name…" bind:value={newVariable} />
    <button
      class="quiet icon addvar"
      onclick={addVariable}
      disabled={!newVariable}
      title="add this variable to the workflow"
      aria-label="add this variable to the workflow"
    >
      <Plus size={14} />
    </button>
    <span></span>
  {/if}
</div>

<style>
  /* The field, and beside it the asset browser when it holds an asset: */
  .field {
    display: flex;
    gap: var(--space-1);
    align-items: start;
    min-width: 0;
  }
  .field > :global(:first-child) {
    flex: 1;
  }
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
  .addvar {
    justify-self: start;
  }
</style>
