<script lang="ts">
  import { PROMPT, reference } from '../references'
  import { Plus, Trash2 } from '@lucide/svelte'
  import ExpandingText from './ExpandingText.svelte'
  import {
    coerce,
    displayValue,
    isReference,
    presetPlaceholder,
    presetView,
    widgetFor,
  } from '../editor'
  import { promptLibrary } from '../promptlib.svelte'
  import { promptTooltip } from '../prompts'
  import { referenceCompletions } from '../refcomplete.svelte'
  import ReferenceField from '../ui/ReferenceField.svelte'
  import type { FieldKind } from '../referenceField'
  import { ASSET, OUTPUT, referenceName } from '../references'
  import { attentionBackends } from '../attentionBackends.svelte'
  import type { Finding } from '../findings'
  import { undoToast } from '../destructive'
  import { OPEN_DOCUMENT } from '../editorShell.svelte'
  import { notify } from '../toast'

  let {
    mode,
    variables = $bindable(),
    overrides = $bindable({}),
    idPrefix,
    backendVariables = new Set<string>(),
    requiredVariables = new Set<string>(),
    constraints = {},
    presets = undefined,
    findings = {},
    kinds = {},
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
    /** Variables with no default: marked required, and the field says so
     * where a default would otherwise show. */
    requiredVariables?: Set<string>
    /** Each variable's declared rules in words (`constraintHints`), shown
     * muted under its field. They state the rule; the server judges it. */
    constraints?: Record<string, string[]>
    /** The definition's `presets` ({selector: {bundle: {variable: value}}}).
     * In override mode a selector is a dropdown of its bundles and the
     * chosen bundle's values are the placeholders of the fields it sets;
     * they are never copied into the overrides. */
    presets?: unknown
    /** What a validate answer said about each variable, placed by its
     * `arguments.<name>` / `variables.<name>` path (`placeFindings`) and
     * shown under the field as text. */
    findings?: Record<string, Finding[]>
    /** The kind of file each variable takes (the catalog's `inputs`): in
     * override mode such a variable is a ReferenceField of that kind */
    kinds?: Record<string, FieldKind>
  } = $props()

  const keys = $derived(Object.keys(variables ?? {}))

  const presetDefinition = $derived({ variables, presets })
  const selectors = $derived(presetView(presetDefinition))

  /** A selector is a dropdown in override mode when its declared default
   * and its override are both plain names - a reference or a non-string
   * stays the text field. */
  function isPresetSelect(key: string): boolean {
    if (mode !== 'override' || !Object.hasOwn(selectors, key)) return false
    const declared = variables[key]
    const plainDefault =
      declared === undefined ||
      declared === null ||
      (typeof declared === 'string' && !isReference(declared))
    const current = overrides[key] ?? ''
    return plainDefault && typeof current === 'string' && !isReference(current)
  }

  function placeholderFor(key: string): string {
    if (mode !== 'override') return ''
    if (requiredVariables.has(key)) return 'required'
    return (
      presetPlaceholder(presetDefinition, overrides, key) ??
      displayValue(variables[key])
    )
  }

  /** Whether a variable is a file field: it takes a kind of file (override
   * mode), or holds an asset: or output: reference */
  function isFileField(key: string): boolean {
    if (mode === 'override' && Object.hasOwn(kinds, key)) return true
    const value = effective(key)
    return (
      referenceName(value, ASSET) !== null ||
      referenceName(value, OUTPUT) !== null
    )
  }

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
      v === undefined ||
      v === null ||
      (typeof v === 'string' && !isReference(v))
    return plain(variables[key]) && (mode === 'define' || plain(overrides[key]))
  }

  /** Define mode: no backend is null, which the engine skips - an empty
   * string would reach diffusers as a backend name. Override mode keeps ''
   * meaning "use the workflow default". */
  function commitBackend(key: string, raw: string) {
    if (mode === 'define' && raw === '') variables[key] = null
    else commit(key, raw)
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

  /** Removed at once, with an Undo that puts it back at its place - before
   * the first key that followed it and is still there. A key added again
   * meanwhile is the person's newer word, so it is left as it is */
  function removeVariable(key: string) {
    const order = Object.keys(variables)
    const value = $state.snapshot(variables[key])
    variables = Object.fromEntries(
      Object.entries(variables).filter(([k]) => k !== key),
    )
    undoToast(
      `Removed variable ${key}`,
      () => {
        if (Object.hasOwn(variables, key))
          return notify.success(`${key} is already back`)
        const entries = Object.entries(variables)
        const after = new Set(order.slice(order.indexOf(key) + 1))
        const found = entries.findIndex(([k]) => after.has(k))
        const at = found < 0 ? entries.length : found
        entries.splice(at, 0, [key, value])
        variables = Object.fromEntries(entries)
      },
      OPEN_DOCUMENT,
    )
  }
</script>

<div class="formgrid top vars" class:withaction={mode === 'define'}>
  {#each keys as key (key)}
    <label for={idPrefix + key}>
      {key}{#if requiredVariables.has(key)}
        <span class="required">required</span>{/if}
    </label>
    <div class="field">
      {#if isPresetSelect(key)}
        {@const current = shown(key)}
        <select
          id={idPrefix + key}
          value={current}
          onchange={(e) => commit(key, e.currentTarget.value)}
        >
          <option value="">
            default ({displayValue(variables[key]) || 'none'})
          </option>
          {#if current && !selectors[key].bundles.includes(current)}
            <option value={current}>{current} (not a preset)</option>
          {/if}
          {#each selectors[key].bundles as bundle (bundle)}
            <option value={bundle}>{bundle}</option>
          {/each}
        </select>
      {:else if isBackendSelect(key)}
        {@const current = shown(key)}
        <select
          id={idPrefix + key}
          value={current}
          onchange={(e) => commitBackend(key, e.currentTarget.value)}
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
      {:else if isFileField(key)}
        <ReferenceField
          id={idPrefix + key}
          value={shown(key)}
          kind={mode === 'override' ? (kinds[key] ?? 'file') : undefined}
          placeholder={placeholderFor(key) || undefined}
          oninput={(raw) => draft(key, raw)}
          onchange={(raw) => commit(key, raw)}
        />
      {:else}
        <ExpandingText
          id={idPrefix + key}
          value={shown(key)}
          placeholder={placeholderFor(key)}
          refStyle={isReference(effective(key))}
          complete={(text) => referenceCompletions(text)}
          title={promptTooltip(effective(key), promptLibrary.texts)}
          oninput={(raw) => draft(key, raw)}
          onchange={(raw) => commit(key, raw)}
          onpromptpick={(name) => commit(key, reference(PROMPT, name))}
        />
      {/if}
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
    {#if Object.hasOwn(selectors, key)}
      <ul class="hints" data-presets-for={key}>
        {#if selectors[key].sets.length}
          <li>sets {selectors[key].sets.join(', ')}</li>
        {/if}
        {#if mode === 'define' && selectors[key].bundles.length}
          <li>bundles: {selectors[key].bundles.join(', ')}</li>
        {/if}
      </ul>
    {/if}
    {#if constraints[key]?.length}
      <ul class="hints" data-hints-for={key}>
        {#each constraints[key] as hint, i (i)}
          <li>{hint}</li>
        {/each}
      </ul>
    {/if}
    {#if findings[key]?.length}
      <ul class="hints findings" data-findings-for={key}>
        {#each findings[key] as finding, i (i)}
          <li class={finding.tone}>{finding.text}</li>
        {/each}
      </ul>
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
  /* The field: */
  .field {
    display: flex;
    gap: var(--space-1);
    align-items: start;
    min-width: 0;
  }
  .field > :global(:first-child) {
    flex: 1;
  }
  .required {
    font-weight: 400;
    font-size: 0.8em;
    color: var(--warn);
  }
  /* Under the field, in the field's column; spans the row when narrow */
  .hints {
    grid-column: 2 / -1;
    margin: calc(-1 * var(--space-1)) 0 0;
    padding: 0;
    list-style: none;
    font-size: 0.85em;
    color: var(--muted);
    overflow-wrap: anywhere;
  }
  @container (max-width: 400px) {
    .hints {
      grid-column: 1 / -1;
    }
  }
  .addvar {
    justify-self: start;
  }
</style>
