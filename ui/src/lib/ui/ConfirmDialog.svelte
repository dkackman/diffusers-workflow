<script lang="ts">
  import { AlertDialog } from 'bits-ui'
  import { confirmState, resolveChoice } from '../confirm.svelte'
  import { holdLayer } from './layers.svelte'

  $effect(() => {
    if (confirmState.open) return holdLayer()
  })
</script>

<!-- Escape and an outside click answer null (Bits reports each as
     onOpenChange(false)); the choice whose value is 'cancel' answers
     'cancel'. Any other choice answers its own value. -->
<AlertDialog.Root
  open={confirmState.open}
  onOpenChange={(open) => {
    if (!open) resolveChoice(null)
  }}
>
  <AlertDialog.Portal>
    <AlertDialog.Overlay class="ui-overlay" />
    <AlertDialog.Content
      class="ui-sheet panel"
      aria-label={confirmState.title}
      interactOutsideBehavior="close"
    >
      <AlertDialog.Description class="ui-message">
        {confirmState.message}
      </AlertDialog.Description>
      <div class="ui-actions">
        {#each confirmState.choices as choice (choice.value)}
          {#if choice.value === 'cancel'}
            <AlertDialog.Cancel class="quiet">
              {choice.label}
            </AlertDialog.Cancel>
          {:else}
            <AlertDialog.Action
              class={choice.danger ? 'danger' : undefined}
              onclick={() => resolveChoice(choice.value)}
            >
              {choice.label}
            </AlertDialog.Action>
          {/if}
        {/each}
      </div>
    </AlertDialog.Content>
  </AlertDialog.Portal>
</AlertDialog.Root>
