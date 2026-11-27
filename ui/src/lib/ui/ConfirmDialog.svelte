<script lang="ts">
  import { AlertDialog } from 'bits-ui'
  import { confirmState, resolveChoice } from '../confirm.svelte'
  import { holdLayer, portalTarget } from './layers.svelte'

  const uid = $props.id()
  const portal = portalTarget(() => confirmState.open)

  $effect(() => {
    if (confirmState.open) return holdLayer()
  })

  // A name dialog's field: why the name will not do, or null
  const invalid = $derived(
    confirmState.input
      ? confirmState.input.validate(confirmState.input.value)
      : null,
  )

  // A name dialog starts in its field, the text selected to type over
  function startHere(field: HTMLInputElement) {
    const timer = setTimeout(() => {
      field.focus()
      field.select()
    })
    return { destroy: () => clearTimeout(timer) }
  }

  function enter(event: KeyboardEvent) {
    if (event.key !== 'Enter') return
    event.preventDefault()
    if (!invalid) resolveChoice('confirm')
  }
</script>

<!-- Escape and an outside click answer null (Bits reports each as
     onOpenChange(false)); the choice whose value is 'cancel' answers
     'cancel'. Any other choice answers its own value; a name dialog's
     confirm is disabled while its name does not validate. `open` is bound
     so Bits reads it back after a dismissal: with another dialog queued it
     is still true, and the dialog stays up for the next question. -->
<AlertDialog.Root
  bind:open={
    () => confirmState.open,
    (open) => {
      if (!open) resolveChoice(null)
    }
  }
>
  <AlertDialog.Portal to={portal.to}>
    <AlertDialog.Overlay class="ui-overlay ui-ask" />
    <AlertDialog.Content
      class="ui-sheet panel ui-ask"
      interactOutsideBehavior="close"
      onOpenAutoFocus={(e) => {
        if (confirmState.input) e.preventDefault()
      }}
    >
      <AlertDialog.Title class="ui-title">
        {confirmState.title}
      </AlertDialog.Title>
      <AlertDialog.Description class="ui-message">
        {confirmState.message}
      </AlertDialog.Description>
      {#if confirmState.input}
        {#key confirmState.input}
          <label class="ui-field">
            <span>{confirmState.input.label}</span>
            <input
              type="text"
              spellcheck="false"
              autocomplete="off"
              bind:value={confirmState.input.value}
              aria-invalid={invalid ? 'true' : undefined}
              aria-describedby={invalid ? `${uid}-why` : undefined}
              onkeydown={enter}
              use:startHere
            />
          </label>
          {#if invalid}
            <p class="ui-why" id="{uid}-why">{invalid}</p>
          {/if}
        {/key}
      {/if}
      <div class="ui-actions">
        {#each confirmState.choices as choice (choice.value)}
          {#if choice.value === 'cancel'}
            <AlertDialog.Cancel class="quiet" title={choice.title}>
              {choice.label}
            </AlertDialog.Cancel>
          {:else}
            <AlertDialog.Action
              class={choice.danger ? 'danger' : undefined}
              title={choice.title}
              disabled={!!invalid}
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

<style>
  :global(.ui-title) {
    margin: 0 0 var(--space-2);
    font-size: var(--t-md, 1rem);
    font-weight: 600;
  }
  .ui-field {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    margin: 0 0 var(--space-2);
  }
  .ui-field input {
    font-family: var(--font-mono);
  }
  .ui-why {
    margin: 0 0 var(--space-2);
    color: var(--bad);
    font-size: var(--t-sm);
  }
</style>
