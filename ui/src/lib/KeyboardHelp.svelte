<script lang="ts">
  import Modal from './ui/Modal.svelte'

  let { open = $bindable(false) }: { open?: boolean } = $props()
</script>

<Modal bind:open label="keyboard shortcuts" wide>
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
</Modal>

<style>
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
    font-family: var(--font-mono);
    font-size: 0.8rem;
    border: 1px solid var(--line);
    border-bottom-width: 2px;
    border-radius: 4px;
    padding: 0.05rem 0.4rem;
    background: var(--panel-2);
  }
</style>
