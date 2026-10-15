<script lang="ts">
  import { isRemembered, remember } from '../expandedNotes'
  import Markdown from './Markdown.svelte'

  // Text that may run long - a note, a description, a docstring - shown at
  // most `lines` lines tall until the reader expands it, so one long entry
  // does not push the rest of the page out of view. The clamp is a max-height
  // with a fade, not -webkit-line-clamp, so it also holds for Markdown's block
  // children. The toggle shows only when the clamp actually hides text,
  // measured while collapsed and again on a resize. Text starts as the state
  // remembered for its rememberKey (expanded or not), else collapsed
  let {
    text,
    lines = 4,
    rememberKey,
    markdown = false,
  }: {
    text: string
    lines?: number
    rememberKey?: string
    markdown?: boolean
  } = $props()

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
    class:plain={!markdown}
    style:--lines={lines}
    bind:this={el}
  >
    {#if markdown}<Markdown source={text} />{:else}{text}{/if}
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
    line-height: 1.5;
    overflow-wrap: anywhere;
  }
  .plain {
    white-space: pre-wrap;
  }
  /* A height clamp, not -webkit-line-clamp: that one does not reliably
     clamp across block children, and a Markdown note is paragraphs and
     lists. The fade says there is more */
  .clamped {
    max-height: calc(var(--lines) * 1.5em);
    overflow: hidden;
    mask-image: linear-gradient(to bottom, #000 70%, transparent);
  }
  .more {
    font-size: var(--t-sm);
    padding: 0;
  }
</style>
