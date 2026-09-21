<script lang="ts">
  import { route } from './router.svelte'
  import { serverHref, sharedHref, wsHref } from './routes'

  // group / section, each a link; the section's rest (a workflow or job
  // name) is the page's own h1, not the crumb's
  const crumbs = $derived.by(() => {
    const view = route.view
    if (view.kind === 'ws')
      return [
        { label: view.workspace, href: wsHref(view.workspace, 'overview') },
        { label: view.section, href: wsHref(view.workspace, view.section) },
      ]
    if (view.kind === 'shared')
      return [
        { label: 'shared', href: sharedHref('prompts') },
        { label: view.section, href: sharedHref(view.section) },
      ]
    return [
      { label: 'server', href: serverHref('status') },
      { label: view.section, href: serverHref(view.section) },
    ]
  })
</script>

<nav class="crumbs" aria-label="breadcrumb">
  {#each crumbs as crumb, i (crumb.href)}
    {#if i > 0}<span class="sep muted">/</span>{/if}
    <a class="plain" href={crumb.href}>{crumb.label}</a>
  {/each}
</nav>

<style>
  .crumbs {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-family: var(--font-mono);
    font-size: var(--t-sm);
  }
  .crumbs a {
    color: var(--muted);
  }
  .crumbs a:last-child {
    color: var(--ink);
  }
</style>
