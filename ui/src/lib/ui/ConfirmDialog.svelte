<script lang="ts">
  import { AlertDialog } from 'bits-ui'
  import { confirmState, resolveConfirm } from '../confirm.svelte'
  import { holdLayer } from './layers.svelte'

  $effect(() => {
    if (confirmState.open) return holdLayer()
  })
</script>

<!-- Escape, an outside click and Cancel all answer false; Bits reports
     each as onOpenChange(false). The confirm button answers true. -->
<AlertDialog.Root
  open={confirmState.open}
  onOpenChange={(open) => {
    if (!open) resolveConfirm(false)
  }}
>
  <AlertDialog.Portal>
    <AlertDialog.Overlay class="ui-overlay" />
    <AlertDialog.Content
      class="ui-sheet panel"
      aria-label="confirm"
      interactOutsideBehavior="close"
    >
      <AlertDialog.Description class="ui-message">
        {confirmState.message}
      </AlertDialog.Description>
      <div class="ui-actions">
        <AlertDialog.Cancel class="quiet"
          >{confirmState.cancelLabel}</AlertDialog.Cancel
        >
        <AlertDialog.Action onclick={() => resolveConfirm(true)}>
          {confirmState.confirmLabel}
        </AlertDialog.Action>
      </div>
    </AlertDialog.Content>
  </AlertDialog.Portal>
</AlertDialog.Root>
