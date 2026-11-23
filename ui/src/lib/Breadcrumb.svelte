<script lang="ts">
  import { fade } from 'svelte/transition'
  import { prefersReducedMotion } from 'svelte/motion'
  import { route } from './router.svelte'
  import {
    SECTION_LABELS,
    sectionLabel,
    serverHref,
    sharedHref,
    wsHref,
  } from './routes'
  import { wsColor } from './wsColor'

  // group / section, each a link; the section's rest (a workflow or job
  // name) is the page's own h1, not the crumb's
  const crumbs = $derived.by(
    (): { label: string; href: string; dot?: string }[] => {
      const view = route.view
      if (view.kind === 'ws')
        return [
          {
            label: view.workspace,
            href: wsHref(view.workspace, 'overview'),
            dot: wsColor(view.workspace),
          },
          {
            label: sectionLabel(view),
            href: wsHref(view.workspace, view.section),
          },
        ]
      if (view.kind === 'shared')
        return [
          { label: SECTION_LABELS.groups.shared, href: sharedHref('prompts') },
          { label: sectionLabel(view), href: sharedHref(view.section) },
        ]
      if (view.kind === 'server')
        return [
          { label: SECTION_LABELS.groups.server, href: serverHref('status') },
          { label: sectionLabel(view), href: serverHref(view.section) },
        ]
      return [{ label: sectionLabel(view), href: '#' + view.path }]
    },
  )
</script>

<nav class="crumbs" aria-label="breadcrumb">
  <!-- keyed by position: on an overview route both crumbs share an href -->
  {#each crumbs as crumb, i (i)}
    {#if i > 0}<span class="sep muted">/</span>{/if}
    <!-- the group crumb fades in when it changes, so a workspace switch
         shows up here as well as in the sidebar; in only, so the row never
         holds two labels at once -->
    {#key crumb.label}
      <a
        class="plain"
        href={crumb.href}
        in:fade={{ duration: prefersReducedMotion.current ? 0 : 150 }}
      >
        {#if crumb.dot}<span class="dot" style:background={crumb.dot}
          ></span>{/if}{crumb.label}
      </a>
    {/key}
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
  .crumbs a {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
  }
  /* the workspace's identity dot, as in the sidebar */
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }
  .crumbs a:last-child {
    color: var(--ink);
  }
</style>
