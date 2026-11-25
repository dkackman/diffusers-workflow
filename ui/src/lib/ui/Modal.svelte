<script lang="ts">
  import type { Snippet } from 'svelte'
  import { Dialog } from 'bits-ui'
  import { holdLayer, portalTarget } from './layers.svelte'

  let {
    open = $bindable(false),
    label,
    size,
    onkeydowncapture,
    onopenfocus,
    children,
  }: {
    open?: boolean
    label: string
    /** `wide` for a longer message; `large`, most of the viewport, for a
     * browsing grid; `fill`, the whole window, for one file shown large
     * (MediaViewer; its look is `.ui-sheet-fill` in patterns.css, app.css
     * being full). A message's width when unset */
    size?: 'wide' | 'large' | 'fill'
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
  const portal = portalTarget(() => open)

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
  <Dialog.Portal to={portal.to}>
    <Dialog.Overlay class="ui-overlay" />
    <Dialog.Content
      class={['ui-sheet panel', size && `ui-sheet-${size}`]
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
