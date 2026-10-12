<script lang="ts">
  import { Combobox } from 'bits-ui'
  import type { SuggestItem } from './suggest'

  let {
    value = $bindable(''),
    suggestions = [],
    complete = undefined,
    openOnFocus = false,
    onchange = undefined,
    oninput = undefined,
    ...rest
  }: {
    value?: string
    suggestions?: readonly string[]
    /** Rows for what has been typed so far, in place of `suggestions` -
     * or null to fall back on them. How a field offers a library once a
     * reference prefix starts it, searched and shown with thumbnails */
    complete?: (text: string) => SuggestItem[] | null
    /** Show the list as soon as the field is focused, for a field that
     * exists to search (the stored-prompt picker); others wait for typing,
     * so tabbing through a form never pops a list over each field */
    openOnFocus?: boolean
    /** The committed value: a chosen suggestion, or typed text on change */
    onchange?: (value: string) => void
    /** Each keystroke's text, for a caller that drafts as it goes */
    oninput?: (value: string) => void
  } & Omit<
    Combobox.InputProps,
    'value' | 'defaultValue' | 'onchange' | 'oninput'
  > = $props()

  // A long class list is offered a page at a time; typing narrows it
  const MAX_SHOWN = 50

  // The completion's rows first, then the plain suggestions that match and
  // that it did not already offer
  function matching(text: string): SuggestItem[] {
    const completed = complete?.(text) ?? []
    const offered = new Set(completed.map((item) => item.value))
    const query = text.toLowerCase()
    const plain = suggestions
      .filter((s) => !offered.has(s) && s.toLowerCase().includes(query))
      .map((value) => ({ value }))
    return [...completed, ...plain].slice(0, MAX_SHOWN)
  }

  let open = $state(false)
  const shown = $derived(matching(value ?? ''))
  // Bits is a select: it keeps the row it last took, so taking that row
  // again would deselect it, and it takes its highlighted row on Enter.
  // Here a suggestion is an offer - each pick is fresh, and Enter takes a
  // row only when the user arrowed to it; otherwise the typed text stands
  let picked = $state('')
  let navigated = false
  // Typed text whose rows had not arrived yet - a library still loading on
  // the first keystroke of a reference. When they land, the list opens
  // for it rather than waiting for the next keystroke
  let awaiting = $state(false)
  $effect(() => {
    if (awaiting && shown.length) {
      open = true
      awaiting = false
    }
  })
</script>

<!-- Free text with suggestions, as a <datalist> offered: what is typed is
     the value, and a suggestion is an offer that replaces it when chosen.
     Nothing matching, nothing shown. -->
<Combobox.Root
  type="single"
  bind:open
  bind:value={picked}
  inputValue={value ?? ''}
  onValueChange={(chosen) => {
    if (!chosen) return
    // Read before the value changes, which recomputes the rows
    const continues = shown.find((item) => item.value === chosen)?.continues
    picked = ''
    value = chosen
    if (continues) {
      // A prefix: still typing, so a draft rather than a commit, and the
      // list reopens on what that prefix offers once Bits has closed it
      oninput?.(chosen)
      navigated = false
      setTimeout(() => {
        if (shown.length) open = true
        else awaiting = true
      })
      return
    }
    onchange?.(chosen)
  }}
>
  <Combobox.Input
    {...rest}
    autocomplete="off"
    oninput={(e) => {
      value = e.currentTarget.value
      navigated = false
      oninput?.(value)
      open = matching(value).length > 0
      awaiting = !open && !!complete?.(value)
    }}
    onblur={() => (awaiting = false)}
    onfocus={() => {
      if (openOnFocus && shown.length) open = true
    }}
    onkeydown={(e) => {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        navigated = true
        // An empty field lists what it can start - the reference prefixes -
        // when asked, rather than on focus
        if (!open && shown.length) open = true
      } else if (e.key === 'Enter' && (!navigated || e.ctrlKey || e.metaKey)) {
        // Prevented here, Bits' own Enter handling is skipped; Ctrl/Cmd+Enter
        // (validate & run) still reaches the page, with the typed text
        e.preventDefault()
        open = false
        onchange?.(e.currentTarget.value)
      }
    }}
    onchange={(e) => onchange?.(e.currentTarget.value)}
  />
  {#if shown.length}
    <Combobox.Portal>
      <Combobox.Content class="ui-suggest panel">
        {#each shown as item (item.value)}
          <Combobox.Item
            value={item.value}
            label={item.value}
            class={item.thumb || item.glyph || item.detail ? 'rich' : ''}
          >
            {#if item.thumb}
              <img class="thumb" src={item.thumb} alt="" loading="lazy" />
            {:else if item.glyph}
              <span class="thumb glyph" aria-hidden="true">{item.glyph}</span>
            {/if}
            <span class="text">
              <span class="name">{item.label ?? item.value}</span>
              {#if item.detail}<span class="detail">{item.detail}</span>{/if}
            </span>
            {#if item.badge}<span class="badge">{item.badge}</span>{/if}
          </Combobox.Item>
        {/each}
      </Combobox.Content>
    </Combobox.Portal>
  {/if}
</Combobox.Root>
