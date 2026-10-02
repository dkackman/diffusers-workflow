<script lang="ts">
  import { Combobox } from 'bits-ui'

  let {
    value = $bindable(''),
    suggestions = [],
    onchange = undefined,
    oninput = undefined,
    ...rest
  }: {
    value?: string
    suggestions?: readonly string[]
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

  function matching(text: string): string[] {
    const query = text.toLowerCase()
    return suggestions
      .filter((s) => s.toLowerCase().includes(query))
      .slice(0, MAX_SHOWN)
  }

  let open = $state(false)
  const shown = $derived(matching(value ?? ''))
</script>

<!-- Free text with suggestions, as a <datalist> offered: what is typed is
     the value, and a suggestion is an offer that replaces it when chosen.
     Nothing matching, nothing shown. -->
<Combobox.Root
  type="single"
  bind:open
  inputValue={value ?? ''}
  onValueChange={(chosen) => {
    if (!chosen) return
    value = chosen
    onchange?.(chosen)
  }}
>
  <Combobox.Input
    {...rest}
    autocomplete="off"
    oninput={(e) => {
      value = e.currentTarget.value
      oninput?.(value)
      open = matching(value).length > 0
    }}
    onchange={(e) => onchange?.(e.currentTarget.value)}
  />
  {#if shown.length}
    <Combobox.Portal>
      <Combobox.Content class="ui-suggest panel">
        {#each shown as suggestion (suggestion)}
          <Combobox.Item value={suggestion} label={suggestion}>
            {suggestion}
          </Combobox.Item>
        {/each}
      </Combobox.Content>
    </Combobox.Portal>
  {/if}
</Combobox.Root>
