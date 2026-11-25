<script lang="ts">
  import { PROMPT, reference } from '../references'
  import { Trash2 } from '@lucide/svelte'
  import { SvelteSet } from 'svelte/reactivity'
  import ExpandingText from './ExpandingText.svelte'
  import VariableNotes from './VariableNotes.svelte'
  import NewVariable from './NewVariable.svelte'
  import {
    displayValue,
    isReference,
    presetPlaceholder,
    presetView,
  } from '../editor'
  import { keepsNull } from './commit'
  import { VARIABLE_TYPES, asType, inferValue, typeOf } from './variableTypes'
  import { isFreeText } from '../workflow/runForm'
  import { promptLibrary } from '../promptlib.svelte'
  import { promptTooltip } from '../prompts'
  import { referenceCompletions } from '../refcomplete.svelte'
  import GroupHeading from '../ui/GroupHeading.svelte'
  import ReferenceField from '../ui/ReferenceField.svelte'
  import type { FieldKind } from '../referenceField'
  import { ASSET, OUTPUT, referenceName } from '../references'
  import { attentionBackends } from '../attentionBackends.svelte'
  import type { Finding } from '../findings'
  import { undoToast } from '../destructive'
  import { OPEN_DOCUMENT } from '../editorShell.svelte'
  import { notify } from '../toast'
  import type { VariableType } from './variableTypes'

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
    advanced = new Set<string>(),
    seed = null,
    locked = new Set<string>(),
    onenhance = undefined,
  }: {
    /** define: edit the workflow's own defaults (add/remove/coerce).
     * override: fill per-run values over read-only defaults. */
    mode: 'define' | 'override'
    // Absent on a workflow that declares none; adding one creates it
    variables?: Record<string, unknown>
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
    /** Variables folded under "Advanced": after the rest, open while one
     * holds an override or a finding */
    advanced?: Set<string>
    /** The variable the workflow's seed names: with no default it says the
     * seed is drawn for each job */
    seed?: string | null
    /** Variables that cannot be edited now (the seed, while a new one is
     * to be drawn): shown, but not typed in */
    locked?: Set<string>
    /** Override mode: a free-text variable offers Enhance…, which asks the
     * page to open the enhancer for this key */
    onenhance?: (key: string) => void
  } = $props()

  const keys = $derived(Object.keys(variables ?? {}))
  const main = $derived(keys.filter((k) => !advanced.has(k)))
  const folded = $derived(keys.filter((k) => advanced.has(k)))
  /** The folded variables holding a value of their own */
  const foldedSet = $derived(folded.filter((k) => overrides[k]).length)
  /** What the person chose for the fold; until they do, it is open while a
   * folded field holds a value or a finding */
  let foldChoice = $state<boolean | null>(null)
  const foldShown = $derived(
    foldChoice ?? folded.some((k) => overrides[k] || findings[k]?.length),
  )

  const presetDefinition = $derived({ variables, presets })
  const selectors = $derived(presetView(presetDefinition))

  /** A selector is a dropdown in override mode when its declared default
   * and its override are both plain names - a reference or a non-string
   * stays the text field. */
  function isPresetSelect(key: string): boolean {
    if (mode !== 'override' || !Object.hasOwn(selectors, key)) return false
    const declared = variables![key]
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
    if (key === seed && variables![key] == null) return 'random each job'
    return (
      presetPlaceholder(presetDefinition, overrides, key) ??
      displayValue(variables![key])
    )
  }

  /** Whether a variable is a file field: it takes a kind of file (override
   * mode), or declares an asset: or output: reference. The widget follows
   * the declared default, never what is being typed, so typing a reference
   * does not swap the input out from under the cursor. */
  function isFileField(key: string): boolean {
    const first = firstKind.get(key)
    // a text field that is still empty may become a file field when the
    // catalog arrives; one holding text is not swapped under it
    if (first !== undefined && (first || overrides[key])) return first
    const is = computeFileField(key)
    firstKind.set(key, is)
    return is
  }
  function computeFileField(key: string): boolean {
    if (mode === 'override' && Object.hasOwn(kinds, key)) return true
    const value = variables![key]
    return (
      referenceName(value, ASSET) !== null ||
      referenceName(value, OUTPUT) !== null
    )
  }
  /** Each field's first-rendered kind, kept until its value is committed:
   * a catalog that loads late must not swap the input out (and the text in
   * it) under a person who is already reading or typing */
  // Not reactive on purpose: it is written while rendering
  // eslint-disable-next-line svelte/prefer-svelte-reactivity
  const firstKind = new Map<string, boolean>()

  /** The default file reference a field would use while it holds nothing,
   * for its tile */
  function defaultFile(key: string): string | undefined {
    const declared = variables![key]
    return mode === 'override' && typeof declared === 'string'
      ? declared
      : undefined
  }

  /** Added and not yet given a value: its first commit infers the type */
  const fresh = new SvelteSet<string>()

  function shown(key: string): string {
    if (mode === 'override') return overrides[key] ?? ''
    return displayValue(variables![key])
  }

  /** The value the field is effectively holding - an override falls back
   * to the default it would leave in place. */
  function effective(key: string): unknown {
    if (mode === 'override') return overrides[key] || variables![key]
    return variables![key]
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
    return (
      plain(variables![key]) && (mode === 'define' || plain(overrides[key]))
    )
  }

  /** Define mode: no backend is null, which the engine skips - an empty
   * string would reach diffusers as a backend name. Override mode keeps ''
   * meaning "use the workflow default". */
  function commitBackend(key: string, raw: string) {
    if (mode === 'define' && raw === '') variables![key] = null
    else commit(key, raw)
  }

  function commit(key: string, raw: string) {
    firstKind.delete(key)
    if (mode === 'override') {
      overrides[key] = raw
      return
    }
    if (keepsNull(raw, variables![key])) return
    // Coerce to the row's type - schema validation runs on the definition
    // itself, so "9" where 9 belongs breaks the workflow
    const type = typeOf(variables![key])
    variables![key] =
      fresh.delete(key) || type === 'none' ? inferValue(raw) : asType(raw, type)
  }

  // Keys whose switch to boolean was refused: their text is not true or false
  let notBoolean = $state<Record<string, boolean>>({})

  function retype(key: string, type: string) {
    fresh.delete(key)
    variables![key] = asType(
      displayValue(variables![key]),
      type as VariableType,
    )
    notBoolean[key] = type === 'boolean' && typeof variables![key] !== 'boolean'
  }

  // An override is live as it is typed, so Run takes what is in the box
  // even before it blurs; a definition commits on change, coerced
  function draft(key: string, raw: string) {
    if (mode === 'override') overrides[key] = raw
  }

  function addVariable(name: string) {
    variables = { ...(variables ?? {}), [name]: '' }
    fresh.add(name)
  }

  /** Removed at once, with an Undo that puts it back at its place - before
   * the first key that followed it and is still there. A key added again
   * meanwhile is the person's newer word, so it is left as it is */
  function removeVariable(key: string) {
    const order = Object.keys(variables!)
    const value = $state.snapshot(variables![key])
    variables = Object.fromEntries(
      Object.entries(variables!).filter(([k]) => k !== key),
    )
    undoToast(
      `Removed variable ${key}`,
      () => {
        if (Object.hasOwn(variables!, key))
          return notify.success(`${key} is already back`)
        const entries = Object.entries(variables!)
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

{#snippet enhance(key: string)}
  {#if mode === 'override' && onenhance && isFreeText(variables![key])}
    <button
      class="quiet"
      onclick={() => onenhance(key)}
      title="Expand an idea into a full prompt with a language model"
    >
      Enhance…
    </button>
  {/if}
{/snippet}

{#snippet row(key: string)}
  <label for={idPrefix + key}>
    {key}{#if requiredVariables.has(key)}
      <span class="required">required</span>{/if}
  </label>
  <div class="field">
    <fieldset class="lock" disabled={locked.has(key)}>
      {#if isPresetSelect(key)}
        {@const current = shown(key)}
        <select
          id={idPrefix + key}
          value={current}
          onchange={(e) => commit(key, e.currentTarget.value)}
        >
          <option value="">
            default ({displayValue(variables![key]) || 'none'})
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
              ? `default (${displayValue(variables![key]) || 'none'})`
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
          fallback={defaultFile(key)}
          title={promptTooltip(effective(key), promptLibrary.texts)}
          oninput={(raw) => draft(key, raw)}
          onchange={(raw) => commit(key, raw)}
        >
          {@render enhance(key)}
        </ReferenceField>
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
        {@render enhance(key)}
      {/if}
      {#if mode === 'define'}
        <select
          class="vartype"
          aria-label="type of {key}"
          value={typeOf(variables![key])}
          onchange={(e) => {
            retype(key, e.currentTarget.value)
            // text that does not fit stays text, which moves nothing
            // the select reads: put it back to what the value is
            e.currentTarget.value = typeOf(variables![key])
          }}
        >
          {#each VARIABLE_TYPES as type (type)}<option>{type}</option>{/each}
        </select>
        {#if notBoolean[key] && typeOf(variables![key]) !== 'boolean'}
          <span class="muted" role="status">
            Only true or false can be a boolean.
          </span>
        {/if}
      {/if}
    </fieldset>
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
  <VariableNotes
    {key}
    {mode}
    selector={selectors[key]}
    hints={constraints[key]}
    findings={findings[key]}
  />
{/snippet}

<div class="formgrid top vars" class:withaction={mode === 'define'}>
  {#each main as key (key)}{@render row(key)}{/each}
  {#if folded.length}
    <div class="fold">
      <GroupHeading
        label="Advanced"
        count={folded.length}
        collapsed={!foldShown}
        ontoggle={() => (foldChoice = !foldShown)}
      >
        {#if !foldShown && foldedSet}<span class="muted">{foldedSet} set</span
          >{/if}
      </GroupHeading>
    </div>
    {#if foldShown}
      {#each folded as key (key)}{@render row(key)}{/each}
    {/if}
  {/if}
  {#if mode === 'define'}
    <NewVariable taken={keys} onadd={addVariable} />
  {/if}
</div>

<style>
  /* The field: */
  .lock {
    display: contents;
    border: 0;
    padding: 0;
    margin: 0;
  }
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
  .fold {
    grid-column: 1 / -1;
  }
  .vartype {
    flex: none;
    width: auto;
    font-size: 0.85em;
  }
</style>
