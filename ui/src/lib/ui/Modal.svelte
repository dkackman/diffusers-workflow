<script lang="ts">
  import type { Snippet } from 'svelte'
  import { Dialog } from 'bits-ui'
  import { holdLayer } from './layers.svelte'

  let {
    open = $bindable(false),
    label,
    wide = false,
    large = false,
    children,
  }: {
    open?: boolean
    label: string
    wide?: boolean
    /** Most of the viewport, for a browsing grid rather than a message */
    large?: boolean
    children: Snippet
  } = $props()

  $effect(() => {
    if (open) return holdLayer()
  })
</script>

<!-- A modal: focus is trapped inside and returns to the trigger on close;
     Escape and an outside click close it -->
<Dialog.Root bind:open>
  <Dialog.Portal>
    <Dialog.Overlay class="ui-overlay" />
    <Dialog.Content
      class={[
        'ui-sheet panel',
        wide && 'ui-sheet-wide',
        large && 'ui-sheet-large',
      ]
        .filter(Boolean)
        .join(' ')}
      aria-label={label}
    >
      {@render children()}
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>
