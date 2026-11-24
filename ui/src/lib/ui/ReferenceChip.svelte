<script lang="ts">
  // A file's reference, the text a workflow argument takes, with a copy
  // button: built by references.ts, never spelled here
  import CopyButton from '../CopyButton.svelte'
  import { reference, type ASSET, type OUTPUT } from '../references'

  let { prefix, name }: { prefix: typeof OUTPUT | typeof ASSET; name: string } =
    $props()

  const text = $derived(reference(prefix, name))
</script>

<span class="refchip">
  <code>{text}</code>
  <CopyButton {text} title="Copy the reference a workflow argument takes" />
</span>

<style>
  .refchip {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    min-width: 0;
  }
  code {
    overflow-wrap: anywhere;
  }
</style>
