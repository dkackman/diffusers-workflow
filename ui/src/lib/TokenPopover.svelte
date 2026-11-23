<script lang="ts">
  import { getApiToken, setApiToken } from './token'
  import { api } from './api'
  import { describeError } from './apiErrors'
  import { notify } from './toast'
  import Popover from './ui/Popover.svelte'

  let {
    open = $bindable(false),
    anchor = null,
  }: { open?: boolean; anchor?: HTMLElement | null } = $props()

  let value = $state(getApiToken())
  let refused = $state<string | null>(null)
  let checking = $state(false)

  // The token is asked about as it is saved: an accepted one ends a lock at
  // once (send() reports the answer, which raises the connection epoch)
  async function save() {
    setApiToken(value)
    value = getApiToken()
    refused = null
    checking = true
    try {
      await api.checkToken()
      open = false
      notify.success('Token accepted')
    } catch (e) {
      const described = describeError(e)
      refused =
        described.kind === 'locked'
          ? 'The server refused that token'
          : described.title
    } finally {
      checking = false
    }
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
      onkeydown={(e) => e.key === 'Enter' && void save()}
    />
    <button onclick={save} disabled={checking}>Save</button>
  </div>
  {#if refused}<p class="refused" role="alert">{refused}</p>{/if}
</Popover>

<style>
  p {
    margin: 0 0 var(--space-3);
  }
  .refused {
    margin: var(--space-2) 0 0;
    color: var(--bad);
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
