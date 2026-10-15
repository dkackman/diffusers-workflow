<script lang="ts">
  import { isRemembered, remember } from '../expandedNotes'

  // Text that may run long - a note, a description, a docstring - shown at
  // most `lines` lines until the reader expands it, so one long entry does
  // not push the rest of the page out of view. The toggle shows only when
  // the clamp actually hides text, measured while collapsed and again on a
  // resize; new text starts collapsed
  let {
    text,
    lines = 4,
    rememberKey,
  }: { text: string; lines?: number; rememberKey?: string } = $props()

  const recalled = () => (rememberKey ? isRemembered(rememberKey) : false)
  let expanded = $state(recalled())
  let overflows = $state(false)
  let el = $state<HTMLElement>()

  // svelte-ignore state_referenced_locally
  let seen = `${rememberKey ?? ''}\u0000${text}`
  $effect(() => {
    const now = `${rememberKey ?? ''}\u0000${text}`
    if (now === seen) return
    seen = now
    expanded = recalled()
  })

  function toggle() {
    expanded = !expanded
    if (rememberKey) remember(rememberKey, expanded)
  }

  $effect(() => {
    void text
    const node = el
    if (!node || expanded) return
    const measure = () =>
      (overflows = node.scrollHeight > node.clientHeight + 1)
    measure()
    // jsdom has no ResizeObserver; the first measurement still stands
    if (typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(measure)
    observer.observe(node)
    return () => observer.disconnect()
  })
</script>

<div class="clamp">
  <div
    class="text"
    class:clamped={!expanded}
    style:--lines={lines}
    bind:this={el}
  >
    {text}
  </div>
  {#if overflows || expanded}
    <button
      type="button"
      class="bare more"
      aria-expanded={expanded}
      onclick={toggle}>{expanded ? 'Show less' : 'Show more'}</button
    >
  {/if}
</div>

<style>
  .clamp {
    min-width: 0;
  }
  .text {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .clamped {
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: var(--lines);
    line-clamp: var(--lines);
    overflow: hidden;
  }
  .more {
    font-size: var(--t-sm);
    padding: 0;
  }
</style>
