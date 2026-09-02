<script lang="ts">
  import { Plus, Trash2 } from '@lucide/svelte'
  import ExpandingText from './ExpandingText.svelte'
  import MediaArgumentInput from './MediaArgumentInput.svelte'
  import {
    classDescription,
    coerce,
    displayValue,
    isReference,
    mediaKindFor,
    mediaLocation,
    widgetFor,
    withMediaLocation,
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

  // Element ids are per instance: two steps both taking 'prompt' would
  // otherwise share id="arg-prompt", and a label click would focus the
  // wrong step's field
  const uid = $props.id()

  let description = $state<PipelineDescription | null>(null)
  let showAll = $state(false)

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
    {@const mediaKind =
      !isReference(args[key]) && !Array.isArray(args[key])
        ? mediaKindFor(parameter, key)
        : null}
    <div class="row">
      <label for={`${uid}-${key}`} title={parameter?.description ?? ''}>
        {key}{#if parameter?.required}<span class="req">*</span>{/if}
      </label>
      <div class="field">
        {#if mediaKind}
          <MediaArgumentInput
            id={`${uid}-${key}`}
            location={mediaLocation(args[key])}
            kind={mediaKind}
            onchange={(location) => {
              args[key] = withMediaLocation(args[key], location)
            }}
          />
        {:else if widget === 'boolean' && !isReference(args[key])}
          <select
            id={`${uid}-${key}`}
            value={String(args[key])}
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
            value={displayValue(args[key], true)}
            onchange={(e) => update(key, e.currentTarget.value)}></textarea>
        {:else if widget === 'textarea'}
          <!-- A reference is never a textarea: widgetFor routes it to the
               input branch below, which carries the datalist and tooltip.
               Prose fields collapse to one line until asked to be a
               document; the picker can swap in a stored prompt. -->
          <ExpandingText
            id={`${uid}-${key}`}
            value={displayValue(args[key])}
            alwaysExpandable
            onchange={(raw) => update(key, raw)}
            onpromptpick={(name) => {
              args[key] = 'prompt:' + name
            }}
          />
        {:else}
          <input
            id={`${uid}-${key}`}
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
                args[parameter.name] = parameter.default ?? ''
              }}
              title={`add ${parameter.name}`}
              aria-label={`add ${parameter.name}`}
            >
              <Plus size={14} />
            </button>
            <span class="availname"
              >{parameter.name}{parameter.required ? ' *' : ''}</span
            >
            {#if parameter.description}
              <span class="muted availdesc">{parameter.description}</span>
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
  .row {
    display: grid;
    grid-template-columns: 170px minmax(0, 1fr) auto;
    gap: 0.7rem;
    align-items: start;
  }
  /* The 170px label track plus an input's intrinsic width needs ~400px;
     under that the label takes its own line above the field */
  @container (max-width: 400px) {
    .row {
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 0.2rem 0.5rem;
    }
    .row > label {
      grid-column: 1 / -1;
      padding-top: 0;
    }
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
