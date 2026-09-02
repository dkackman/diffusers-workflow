<script lang="ts">
  import { focusTrap } from './focusTrap'

  let { open = $bindable(false) }: { open?: boolean } = $props()
</script>

{#if open}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div class="scrim" role="presentation" onclick={() => (open = false)}>
    <div
      class="sheet panel"
      role="dialog"
      aria-label="keyboard shortcuts"
      aria-modal="true"
      tabindex="-1"
      use:focusTrap
      onclick={(e) => e.stopPropagation()}
    >
      <h2>Keyboard shortcuts</h2>
      <dl>
        <dt><kbd>Ctrl/⌘</kbd> + <kbd>S</kbd></dt>
        <dd>save (workflow &amp; prompt editors)</dd>
        <dt><kbd>Ctrl/⌘</kbd> + <kbd>Enter</kbd></dt>
        <dd>validate &amp; run (workflow editor)</dd>
        <dt><kbd>Esc</kbd></dt>
        <dd>close this help, the gallery drawer, and other panels</dd>
        <dt><kbd>?</kbd></dt>
        <dd>show this help</dd>
      </dl>
      <button class="quiet" onclick={() => (open = false)}>close</button>
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
    z-index: 50;
  }
  .sheet {
    min-width: min(420px, 92vw);
    max-width: 480px;
  }
  dl {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: var(--space-2) var(--space-4);
    margin: var(--space-4) 0;
    align-items: baseline;
  }
  dt {
    white-space: nowrap;
  }
  dd {
    margin: 0;
    color: var(--muted);
  }
  kbd {
    font-family: ui-monospace, 'Cascadia Code', monospace;
    font-size: 0.8rem;
    border: 1px solid var(--line);
    border-bottom-width: 2px;
    border-radius: 4px;
    padding: 0.05rem 0.4rem;
    background: var(--panel-2);
  }
</style>
