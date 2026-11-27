<script lang="ts">
  import Suggest from '../ui/Suggest.svelte'
  import { SvelteMap } from 'svelte/reactivity'
  import { PROMPT, reference } from '../references'
  import { Plus, Trash2 } from '@lucide/svelte'
  import ExpandingText from './ExpandingText.svelte'
  import ReferenceField from '../ui/ReferenceField.svelte'
  import type { FieldKind } from '../referenceField'
  import {
    classDescription,
    displayValue,
    isReference,
    widgetFor,
  } from '../editor'
  import { addedArgument, commitArgument, keepsNull } from './commit'
  import {
    mediaKindFor,
    mediaLocation,
    withMediaLocation,
  } from '../mediaArguments'
  import { promptLibrary } from '../promptlib.svelte'
  import { promptTooltip } from '../prompts'
  import { referenceCompletions } from '../refcomplete.svelte'
  import ClampedText from '../ui/ClampedText.svelte'
  import { latestOnly } from '../latest'
  import { sleep } from '../poll'
  import type { PipelineDescription, StepArguments } from '../types'
  import { resource } from '../resource.svelte'

  let {
    args = $bindable(),
    componentType,
    target = 'call',
    hide = [],
    suggestions = [],
    earlier = () => [],
  }: {
    // Absent on a step that sets none; the first one added creates it
    args?: StepArguments
    componentType: string
    target?: 'call' | 'init' | 'load' | 'task'
    hide?: string[]
    suggestions?: readonly string[]
    // The earlier steps' results a media argument of `kind` can take
    earlier?: (kind?: FieldKind) => string[]
  } = $props()

  // Element ids are per instance: two steps both taking 'prompt' would
  // otherwise share id="arg-prompt", and a label click would focus the
  // wrong step's field
  const uid = $props.id()

  let showAll = $state(false)

  // Debounced: the class name arrives keystroke by keystroke, and every
  // prefix would otherwise fire (and cache) a doomed lookup. A class's
  // description never changes while the server runs, so it watches nothing
  const typing = latestOnly()
  const lookup = resource(
    async () => {
      const [wanted, kind] = [componentType, target]
      const current = typing()
      if (!wanted) return null
      await sleep(300)
      if (!current()) return null
      const found = await classDescription(wanted, kind)
      if (found && current()) recommit(found)
      return found
    },
    { watch: [] },
  )
  const description = $derived(
    lookup.status === 'ready' ? (lookup.data ?? null) : null,
  )

  const parameters = $derived(
    new Map((description?.parameters ?? []).map((p) => [p.name, p])),
  )
  const unused = $derived(
    (description?.parameters ?? []).filter(
      (p) => !(p.name in (args ?? {})) && !hide.includes(p.name),
    ),
  )
  const shownKeys = $derived(
    Object.keys(args ?? {}).filter((key) => !hide.includes(key)),
  )

  /** Writes on the person's edit only: a mount that wrote an empty object
   * would mark the document unsaved */
  function write(key: string, value: unknown) {
    if (args) args[key] = value
    else args = { [key]: value }
  }

  function update(key: string, raw: string) {
    const previous = args?.[key]
    write(key, commitArgument(parameters.get(key), raw, previous, key))
    if (!description) holdEarly(key, raw, previous)
  }

  // Edits typed before the signature arrived, committed again by the lookup
  // once it does (#1708). Each holds the arguments object, so it still lands
  // if the editor has closed
  type Early = {
    held: StepArguments
    raw: string
    written: unknown
    original: unknown
  }
  const early = new SvelteMap<string, Early>()

  function holdEarly(key: string, raw: string, previous: unknown) {
    const held = args!
    // a second edit of early text still answers to the value first replaced
    const prior = early.get(key)
    const original =
      prior && prior.written === previous ? prior.original : previous
    early.set(key, { held, raw, written: held[key], original })
  }

  /** A value changed since its edit is left alone */
  function recommit(found: PipelineDescription) {
    for (const [key, { held, raw, written, original }] of early) {
      const parameter = found.parameters.find((p) => p.name === key)
      if (parameter && held[key] === written) {
        held[key] = commitArgument(parameter, raw, original, key)
      }
    }
    early.clear()
  }

  /** A ReferenceField's commit: a media argument holds a location, anything
   * else showing there is a reference being replaced */
  function updateReference(key: string, raw: string, media: boolean) {
    const previous = args?.[key]
    if (media && !keepsNull(raw, previous)) {
      write(key, withMediaLocation(previous, raw))
    } else update(key, raw)
  }

  // Free-form additions belong wherever the schema is open-ended: the
  // callable takes **kwargs, or no schema resolved at all
  const openEnded = $derived(!description || description.accepts_kwargs)
  let customName = $state('')

  function addCustom() {
    const name = customName.trim()
    if (!name || name in (args ?? {})) return
    write(name, addedArgument(name, parameters.get(name)))
    customName = ''
  }

  function remove(key: string) {
    if (args) delete args[key]
  }
</script>

<div class="args">
  {#each shownKeys as key (key)}
    {@const parameter = parameters.get(key)}
    {@const widget = widgetFor(parameter, args?.[key])}
    {@const mediaKind = Array.isArray(args?.[key])
      ? null
      : mediaKindFor(parameter, key)}
    <div class="formgrid withaction top row">
      <label for={`${uid}-${key}`} title={parameter?.description ?? ''}>
        {key}{#if parameter?.required}<span class="req">*</span>{/if}
      </label>
      <div class="field">
        {#if mediaKind || isReference(args?.[key])}
          <ReferenceField
            id={`${uid}-${key}`}
            value={mediaLocation(args?.[key])}
            kind={mediaKind ?? undefined}
            earlier={earlier(mediaKind ?? undefined)}
            {suggestions}
            title={promptTooltip(args?.[key], promptLibrary.texts)}
            onchange={(location) => {
              updateReference(key, location, !!mediaKind)
            }}
          />
        {:else if widget === 'boolean'}
          <select
            id={`${uid}-${key}`}
            value={String(args?.[key])}
            onchange={(e) => update(key, e.currentTarget.value)}
          >
            <option value="true">true</option>
            <option value="false">false</option>
          </select>
        {:else if widget === 'json'}
          <textarea
            id={`${uid}-${key}`}
            spellcheck="false"
            rows="3"
            value={displayValue(args?.[key], true)}
            onchange={(e) => update(key, e.currentTarget.value)}></textarea>
        {:else if widget === 'textarea'}
          <!-- Prose fields collapse to one line until asked to be a
               document; the picker can swap in a stored prompt. -->
          <ExpandingText
            id={`${uid}-${key}`}
            value={displayValue(args?.[key])}
            alwaysExpandable
            title={promptTooltip(args?.[key], promptLibrary.texts)}
            onchange={(raw) => update(key, raw)}
            onpromptpick={(name) => {
              write(key, reference(PROMPT, name))
            }}
          />
        {:else}
          <Suggest
            id={`${uid}-${key}`}
            {suggestions}
            complete={(text) => referenceCompletions(text)}
            title={promptTooltip(args?.[key], promptLibrary.texts)}
            value={displayValue(args?.[key])}
            onchange={(value) => update(key, value)}
          />
        {/if}
        {#if parameter?.description}
          <div class="hint muted">
            <ClampedText text={parameter.description} lines={3} />
          </div>
        {/if}
      </div>
      <button
        class="quiet icon"
        onclick={() => remove(key)}
        title="remove this argument"
        aria-label="remove this argument"
      >
        <Trash2 size={14} />
      </button>
    </div>
  {/each}

  {#if unused.length}
    <button
      class="quiet discover"
      onclick={() => (showAll = !showAll)}
      title="every argument this callable accepts, discovered from its signature"
    >
      {showAll ? 'hide' : 'show'} available arguments ({unused.length})
    </button>
    {#if showAll}
      <div class="available">
        {#each unused as parameter (parameter.name)}
          <div class="availrow">
            <button
              class="quiet icon"
              onclick={() => {
                write(parameter.name, addedArgument(parameter.name, parameter))
              }}
              title={`add ${parameter.name}`}
              aria-label={`add ${parameter.name}`}
            >
              <Plus size={14} />
            </button>
            <span class="availname">
              {parameter.name}{parameter.required ? ' *' : ''}
            </span>
            {#if parameter.description}
              <div class="muted availdesc">
                <ClampedText text={parameter.description} lines={2} />
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
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
  .row.add {
    grid-template-columns: 1fr auto;
    max-width: 340px;
  }
  .req {
    color: var(--warn);
    margin-left: 2px;
  }
  .hint {
    margin-top: 0.15rem;
    max-width: 60ch;
  }
  .icon {
    display: inline-flex;
    align-items: center;
    padding: 0.4rem 0.5rem;
  }
  .discover {
    align-self: start;
    font-size: 0.8rem;
  }
  .available {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    border-left: 2px solid var(--line);
    padding-left: var(--space-2);
  }
  .availrow {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
  }
  .availrow .icon {
    align-self: center;
    flex: none;
  }
  .availname {
    font-weight: 600;
    flex: none;
  }
  .availdesc {
    font-size: 0.75rem;
    overflow-wrap: anywhere;
  }
</style>
