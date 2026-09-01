<script lang="ts">
  import { Copy, CircleCheck } from '@lucide/svelte'
  import { notify } from './toast'

  let {
    text,
    title = 'copy to clipboard',
    class: className = '',
  }: {
    text: string
    title?: string
    class?: string
  } = $props()

  let copied = $state(false)

  async function copy() {
    try {
      await navigator.clipboard.writeText(text)
      copied = true
      setTimeout(() => (copied = false), 1500)
    } catch {
      notify.error('Could not copy to clipboard')
    }
  }
</script>

<button
  class="quiet copybtn {className}"
  class:copied
  onclick={copy}
  {title}
  aria-label={title}
>
  {#if copied}<CircleCheck size={14} />{:else}<Copy size={14} />{/if}
</button>

<style>
  .copybtn {
    padding: 0.3rem;
    line-height: 0;
  }
  .copybtn.copied {
    color: var(--good);
    border-color: var(--good);
  }
</style>
