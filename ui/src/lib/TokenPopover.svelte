<script lang="ts">
  import { getApiToken, setApiToken } from './token'

  let { open = $bindable(false) }: { open?: boolean } = $props()

  let value = $state(getApiToken())
  let saved = $state(false)

  function save() {
    setApiToken(value)
    value = getApiToken()
    saved = true
    setTimeout(() => (saved = false), 1500)
  }

  // Click-anywhere-else closes; the toggle button and the panel itself
  // stop propagation so their clicks never reach this handler
  function onWindowClick() {
    if (open) open = false
  }
</script>

<svelte:window onclick={onWindowClick} />

{#if open}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="pop panel"
    role="dialog"
    aria-label="API token"
    tabindex="-1"
    onclick={(e) => e.stopPropagation()}
  >
    <p class="muted">
      Only needed if the server was started with <code>--token</code> or
      <code>DW_API_TOKEN</code>. Stored in this browser's local storage.
    </p>
    <div class="row">
      <input
        type="password"
        placeholder="API token"
        bind:value
        onkeydown={(e) => e.key === 'Enter' && save()}
      />
      <button onclick={save}>{saved ? 'Saved' : 'Save'}</button>
    </div>
  </div>
{/if}

<style>
  .pop {
    position: absolute;
    top: calc(100% + 4px);
    right: var(--space-4);
    z-index: 30;
    min-width: min(320px, 90vw);
    box-shadow: 0 6px 24px color-mix(in srgb, var(--bg) 60%, transparent);
    font-size: 0.85rem;
  }
  p {
    margin: 0 0 var(--space-3);
  }
  .row {
    display: flex;
    gap: var(--space-2);
  }
  input {
    flex: 1;
    min-width: 0;
  }
</style>
