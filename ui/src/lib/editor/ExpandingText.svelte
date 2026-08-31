<script lang="ts">
  import { Ellipsis, Minimize2 } from '@lucide/svelte'
  import { isLongText } from '../editor'
  import { promptLibrary } from '../promptlib.svelte'

  let {
    id,
    value,
    placeholder = '',
    refStyle = false,
    listId = undefined,
    title = '',
    alwaysExpandable = false,
    onchange,
    oninput = undefined,
    onpromptpick = undefined,
  }: {
    id: string
    value: string
    placeholder?: string
    refStyle?: boolean
    listId?: string
    title?: string
    /** Offer the expand affordance even while the text is still short -
     * for fields that exist to hold documents (prompts). */
    alwaysExpandable?: boolean
    onchange: (raw: string) => void
    oninput?: (raw: string) => void
    /** When set, the expanded view offers the stored-prompt picker. */
    onpromptpick?: (name: string) => void
  } = $props()

  let expanded = $state(false)

  // The affordance appears when there is (or will be) a document here:
  // the value is long, the default it falls back to is long, or the
  // field is one the user writes prose into
  const expandable = $derived(
    alwaysExpandable || isLongText(value) || isLongText(placeholder),
  )

  /** Grow with the content - a document field never scrolls inside
   * itself. Focuses on mount so expanding flows straight into typing. */
  function autosize(node: HTMLTextAreaElement) {
    const resize = () => {
      node.style.height = 'auto'
      node.style.height = `${node.scrollHeight + 2}px`
    }
    resize()
    node.focus()
    node.addEventListener('input', resize)
    return { destroy: () => node.removeEventListener('input', resize) }
  }
</script>

{#if expanded}
  <div class="expwrap">
    <textarea
      {id}
      class:ref={refStyle}
      {title}
      {placeholder}
      spellcheck="true"
      {value}
      use:autosize
      oninput={(e) => oninput?.(e.currentTarget.value)}
      onchange={(e) => onchange(e.currentTarget.value)}></textarea>
    <button
      class="quiet corner"
      onclick={() => (expanded = false)}
      title="collapse to one line"
      aria-label="collapse to one line"
    >
      <Minimize2 size={13} />
    </button>
    {#if onpromptpick && promptLibrary.names?.length}
      <select
        class="promptpick"
        title="replace this text with a stored prompt from the library"
        onchange={(e) => {
          if (!e.currentTarget.value) return
          onpromptpick(e.currentTarget.value)
          e.currentTarget.value = ''
          expanded = false
        }}
      >
        <option value="">use a stored prompt…</option>
        {#each promptLibrary.names ?? [] as promptName (promptName)}
          <option value={promptName}>{promptName}</option>
        {/each}
      </select>
    {/if}
  </div>
{:else}
  <div class="linewrap">
    <input
      {id}
      class:ref={refStyle}
      class:roomy={expandable}
      list={listId}
      autocomplete="off"
      {title}
      {placeholder}
      {value}
      oninput={(e) => oninput?.(e.currentTarget.value)}
      onchange={(e) => onchange(e.currentTarget.value)}
    />
    {#if expandable}
      <button
        class="quiet corner"
        onclick={() => (expanded = true)}
        title="expand to edit as a document"
        aria-label="expand to edit as a document"
      >
        <Ellipsis size={13} />
      </button>
    {/if}
  </div>
{/if}

<style>
  .linewrap,
  .expwrap {
    position: relative;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    min-width: 0;
  }
  input.roomy {
    padding-right: 2rem;
  }
  textarea {
    resize: none;
    overflow: hidden;
    min-height: 9.5rem;
    padding-right: 2rem;
  }
  .corner {
    position: absolute;
    top: 3px;
    right: 3px;
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.3rem;
    border: 0;
    color: var(--muted);
  }
  .corner:hover {
    color: var(--accent);
    filter: none;
  }
  .promptpick {
    align-self: flex-start;
    max-width: 260px;
    font-size: 0.8rem;
  }
</style>
