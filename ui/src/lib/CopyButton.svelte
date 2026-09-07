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

  function fallbackCopy(value: string): boolean {
    const textarea = document.createElement('textarea')
    textarea.value = value
    textarea.style.position = 'fixed'
    textarea.style.opacity = '0'
    document.body.appendChild(textarea)
    textarea.focus()
    textarea.select()
    let ok: boolean
    try {
      ok = document.execCommand('copy')
    } catch {
      ok = false
    }
    document.body.removeChild(textarea)
    return ok
  }

  async function copy() {
    try {
      if (navigator.clipboard) {
        await navigator.clipboard.writeText(text)
      } else if (!fallbackCopy(text)) {
        throw new Error('copy failed')
      }
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
