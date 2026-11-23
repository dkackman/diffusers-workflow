<script module lang="ts">
  import { route } from '../router.svelte'

  // The path the last focus answered. At load it is the path the app opened
  // on, so the first render never takes focus; a page mounted by a
  // navigation (a new path) does. Only the path changing counts: the router
  // replaces `route.parts` on any same-path hashchange too (a sidebar link, a
  // hand-edited query, Back between two queries), and a rename
  // (`renameRoute`) edits it in place; a quiet reload never touches the route
  let seen = route.parts.join('/')
  // The header that named the tab last: one leaving clears the title only
  // while it is still that one, so a page with no header is not left under
  // the name of the page before it
  let naming: object | null = null
</script>

<script lang="ts">
  // The row a page opens with: the back link, the h1 (which names the tab,
  // and takes focus after a navigation so a screen reader starts at the new
  // page), the secondary line, the count, the page's own details
  // (`children`), the filter and the actions. A `nested` header is a
  // section of a page that has its own: an h2 that neither names the tab
  // nor takes focus
  import { tick, untrack, type Snippet } from 'svelte'
  import { refreshTitle } from '../finishAlert'
  import { pageFrame } from '../pageFrame.svelte'

  let {
    title,
    secondary,
    count,
    back,
    actions,
    filter,
    children,
    baseline = false,
    nested = false,
  }: {
    title: string
    secondary?: string
    count?: { n: number; noun: string; plural?: string }
    back?: { href: string; label: string }
    actions?: Snippet
    filter?: Snippet
    children?: Snippet
    baseline?: boolean
    nested?: boolean
  } = $props()

  let h1: HTMLHeadingElement | undefined = $state()

  const me = {}
  // Tracks `title` only: the title it writes is read back by refreshTitle
  const name = (to: string) =>
    untrack(() => {
      pageFrame.title = to
      refreshTitle()
    })
  $effect(() => {
    if (nested) return
    name(title)
    naming = me
    return () => {
      if (naming !== me) return
      naming = null
      name('')
    }
  })

  $effect(() => {
    const parts = route.parts
    const path = parts.join('/')
    if (nested || path === seen) return
    // After the DOM settles: the page this navigation left may still be
    // mounted when its header sees the new path, and must not take focus
    void tick().then(() => {
      if (!h1?.isConnected || parts !== route.parts || path === seen) return
      seen = path
      h1.focus()
    })
  })

  const counted = $derived(
    count
      ? `${count.n} ${count.n === 1 ? count.noun : (count.plural ?? count.noun + 's')}`
      : '',
  )
</script>

<div class="pagehead" class:baseline>
  {#if back}<a class="back plain muted" href={back.href}>← {back.label}</a>{/if}
  {#if nested}<h2>{title}</h2>{:else}<h1 tabindex="-1" bind:this={h1}>
      {title}
    </h1>{/if}
  {#if secondary}<span class="secondary muted">{secondary}</span>{/if}
  {#if count}<span class="count num muted">{counted}</span>{/if}
  {#if children}{@render children()}{/if}
  <span class="flex"></span>
  {#if filter}{@render filter()}{/if}
  {#if actions}{@render actions()}{/if}
</div>

<style>
  h1:focus {
    outline: none;
  }
  /* A long workflow path wraps rather than pushing a phone sideways */
  h1 {
    min-width: 0;
    overflow-wrap: break-word;
  }
  .secondary {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
  }
  .count {
    font-size: var(--t-sm);
  }
  .back {
    font-size: var(--t-sm);
  }
  .pagehead :global(.filter) {
    max-width: 220px;
  }
  /* A link to the page's sibling list, counted ("examples 12") */
  .pagehead :global(.sidelink) {
    font-size: 0.82rem;
    white-space: nowrap;
  }
  .pagehead :global(.sidelink:hover) {
    color: var(--ink);
  }
</style>
