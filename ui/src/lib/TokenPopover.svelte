<script lang="ts">
  import { getApiToken, setApiToken } from './token'
  import Popover from './ui/Popover.svelte'

  let {
    open = $bindable(false),
    anchor = null,
  }: { open?: boolean; anchor?: HTMLElement | null } = $props()

  let value = $state(getApiToken())
  let saved = $state(false)

  function save() {
    setApiToken(value)
    value = getApiToken()
    saved = true
    setTimeout(() => (saved = false), 1500)
  }
</script>

<Popover bind:open label="API token" {anchor} align="end">
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
</Popover>

<style>
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
