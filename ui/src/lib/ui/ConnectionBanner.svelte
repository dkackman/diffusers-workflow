<script lang="ts">
  // Under the header while the server cannot be reached or wants a token:
  // the one place either is said, so a page's own failure need not repeat it
  import { connection } from '../connection.svelte'
  import { checkNow } from '../statusPoll'

  let { ontoken }: { ontoken: () => void } = $props()
</script>

{#if connection.state === 'offline'}
  <div class="banner" role="status">
    <span>Can't reach the server. Retrying every few seconds.</span>
    <button onclick={() => void checkNow()}>Retry now</button>
  </div>
{:else if connection.state === 'locked'}
  <div class="banner" role="status">
    <span>This server needs an API token.</span>
    <button onclick={ontoken}>Enter token</button>
  </div>
{/if}

<style>
  .banner {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2) var(--space-3);
    padding: var(--space-2) 1.2rem;
    border-bottom: 1px solid var(--line);
    background: color-mix(in srgb, var(--bad) 10%, var(--panel));
    font-size: var(--t-sm);
  }
  .banner button {
    padding: 0.2rem 0.6rem;
  }
</style>
