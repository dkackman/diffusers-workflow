<script lang="ts">
  import type { Snippet } from 'svelte'
  import { Dialog } from 'bits-ui'
  import { holdLayer } from './layers.svelte'

  let {
    open = $bindable(false),
    label,
    wide = false,
    large = false,
    fill = false,
    onkeydowncapture,
    onopenfocus,
    children,
  }: {
    open?: boolean
    label: string
    wide?: boolean
    /** Most of the viewport, for a browsing grid rather than a message */
    large?: boolean
    /** The whole window, for one file shown large (MediaViewer); its look
     * is `.ui-sheet-fill` in patterns.css, app.css being full */
    fill?: boolean
    /** Every key pressed inside the sheet, before its target sees it */
    onkeydowncapture?: (event: KeyboardEvent) => void
    /** Where focus goes on open, when not the first control: call
     * preventDefault and focus it. A dialog that spends starts on Cancel */
    onopenfocus?: (event: Event) => void
    children: Snippet
  } = $props()

  $effect(() => {
    if (open) return holdLayer()
  })

  // A press on a toast (its Undo, its close) is not a press outside the
  // sheet: the sheet stays open under it
  function keepForToast(event: PointerEvent) {
    const target = event.target
    if (target instanceof Element && target.closest('[data-sonner-toaster]'))
      event.preventDefault()
  }
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
        fill && 'ui-sheet-fill',
      ]
        .filter(Boolean)
        .join(' ')}
      aria-label={label}
      {onkeydowncapture}
      onOpenAutoFocus={onopenfocus}
      onInteractOutside={keepForToast}
    >
      {@render children()}
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>
