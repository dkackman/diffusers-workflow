<script lang="ts">
  import type { Snippet } from 'svelte'
  import { Dialog } from 'bits-ui'
  import { holdLayer } from './layers.svelte'

  let {
    open = $bindable(false),
    label,
    wide = false,
    children,
  }: {
    open?: boolean
    label: string
    wide?: boolean
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
      class={wide ? 'ui-sheet ui-sheet-wide panel' : 'ui-sheet panel'}
      aria-label={label}
    >
      {@render children()}
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>
