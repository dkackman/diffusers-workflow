<script lang="ts">
  import { confirmState, resolveConfirm } from './confirm.svelte'
  import { focusTrap } from './focusTrap'

  function onKeydown(event: KeyboardEvent) {
    if (confirmState.open && event.key === 'Escape') {
      event.preventDefault()
      resolveConfirm(false)
    }
  }
</script>

<svelte:window onkeydown={onKeydown} />

{#if confirmState.open}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div class="scrim" role="presentation" onclick={() => resolveConfirm(false)}>
    <div
      class="sheet panel"
      role="alertdialog"
      aria-modal="true"
      aria-label="confirm"
      tabindex="-1"
      use:focusTrap
      onclick={(e) => e.stopPropagation()}
    >
      <p>{confirmState.message}</p>
      <div class="actions">
        <button class="quiet" onclick={() => resolveConfirm(false)}
          >{confirmState.cancelLabel}</button
        >
        <button onclick={() => resolveConfirm(true)}
          >{confirmState.confirmLabel}</button
        >
      </div>
    </div>
  </div>
{/if}

<style>
  .scrim {
    position: fixed;
    inset: 0;
    background: color-mix(in srgb, var(--bg) 65%, transparent);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 60;
  }
  .sheet {
    min-width: min(360px, 92vw);
    max-width: 440px;
  }
  p {
    margin: 0 0 var(--space-4);
    white-space: pre-line;
  }
  .actions {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-2);
  }
</style>
