<script lang="ts">
  import { untrack } from 'svelte'
  import {
    ChevronDown,
    ChevronRight,
    Eye,
    PanelLeftClose,
    PanelLeftOpen,
    Plus,
  } from '@lucide/svelte'
  import { slide } from 'svelte/transition'
  import { prefersReducedMotion } from 'svelte/motion'
  import { liveQuery, route } from './router.svelte'
  import { SERVER_ITEMS, SHARED_ITEMS, WS_ITEMS } from './navItems'
  import {
    SECTION_LABELS,
    serverHref,
    sharedHref,
    switchHref,
    wsHref,
    type SharedSection,
  } from './routes'
  import { resource } from './resource.svelte'
  import { relistWorkspaces, workspace } from './workspace.svelte'
  import NewWorkspace from './NewWorkspace.svelte'
  import { formatBytes } from './format'
  import { wsColor } from './wsColor'
  import { groupWorkspaces, splitName } from './wsGroups'
  import { hiddenWorkspaces } from './hiddenWorkspaces.svelte'
  import { storageGet, storageSet } from './storage'

  let {
    collapsed,
    onToggle,
    inert = false,
  }: { collapsed: boolean; onToggle: () => void; inert?: boolean } = $props()

  // A workspace an agent creates or deletes shows within one feed interval:
  // the counter is server-wide, so the open workspace's feed carries it.
  // What the listing holds is module state (workspace.svelte.ts), drawn
  // from there; the resource is how it stays fresh
  resource(() => relistWorkspaces(), { watch: ['workspaces'] })
  // The workspace whose sections are open: the route's when it names one,
  // else the one the UI is scoped to (a shared or server page still sits
  // inside a workspace's context)
  const expanded = $derived(
    route.view.kind === 'ws' ? route.view.workspace : workspace.current,
  )
  // A switch is shown, not just repainted: the new workspace's sections
  // slide open where the old ones slid shut, and its name row settles from
  // a stronger --select tint to the block's resting one. The first render
  // is not a switch, so the flash waits for `expanded` to change. Both
  // honour the reduced-motion preference the way `.pulse-dot` does.
  let switched = $state(false)
  // the initial value on purpose: it is what a later change is compared to
  let seen = untrack(() => expanded)
  $effect(() => {
    if (expanded !== seen) {
      seen = expanded
      switched = true
    }
  })
  const motion = $derived({ duration: prefersReducedMotion.current ? 0 : 150 })
  const wsSection = $derived(
    route.view.kind === 'ws' ? route.view.section : null,
  )
  const sharedSection = $derived(
    route.view.kind === 'shared' ? route.view.section : null,
  )
  const serverSection = $derived(
    route.view.kind === 'server' ? route.view.section : null,
  )

  // prompt-edit sits under Prompts in the nav
  const sharedActive = (s: SharedSection) =>
    sharedSection === s || (s === 'prompts' && sharedSection === 'prompt-edit')

  const folded = $derived(
    groupWorkspaces(workspace.names ?? [workspace.current], {
      hidden: hiddenWorkspaces.names,
      open: workspace.current,
    }),
  )
  let openGroups = $state(storageGet<string[]>('open-ws-groups', []))
  function toggleGroup(prefix: string) {
    openGroups = openGroups.includes(prefix)
      ? openGroups.filter((p) => p !== prefix)
      : [...openGroups, prefix]
    storageSet('open-ws-groups', openGroups)
  }
  let showHidden = $state(false)

  let adding = $state(false)
  // held here, not in the form, so collapsing the rail keeps a typed name
  let draft = $state('')
</script>

<!-- under a group header the shared prefix is read out, not drawn: it would
     take the width that tells the rows apart -->
{#snippet wsName(name: string, prefix?: string)}
  {@const { said, own } = splitName(name, prefix)}
  <span class="name"
    >{#if said}<span class="sr-only">{said}</span>{/if}{own}</span
  >
{/snippet}

{#snippet wsRow(name: string, prefix?: string)}
  {#if name === expanded}
    <!-- The open workspace is one block: a --select rail and tint
               behind its name and sections, a guide line down its children,
               so siblings read as outside it -->
    <div class="ws open" class:flash={switched} style:--ws-dot={wsColor(name)}>
      {#if collapsed}
        <span class="dot solo" title={name}></span>
      {:else}
        <span class="wsname" title={name}>
          <ChevronDown size={13} class="chev" />
          <span class="dot"></span>
          {@render wsName(name, prefix)}
          {#if workspace.usage[name]}
            <span
              class="num muted size"
              title="{workspace.usage[name].files} files"
            >
              {formatBytes(workspace.usage[name].bytes)}
            </span>
          {/if}
        </span>
      {/if}
      <div class="sections" transition:slide={motion}>
        {#each WS_ITEMS as item (item.section)}
          <a
            class="plain item"
            href={wsHref(name, item.section)}
            aria-current={wsSection === item.section ? 'page' : undefined}
            title={item.label}
          >
            <item.icon size={15} />{#if !collapsed}<span class="label">
                {item.label}
              </span>{/if}
          </a>
        {/each}
      </div>
    </div>
  {:else if collapsed}
    <a
      class="plain dotlink"
      href={switchHref(name, route.view, liveQuery.now)}
      title={name}
      aria-label="{name} workspace"
      style:--ws-dot={wsColor(name)}
    >
      <span class="dot solo"></span>
    </a>
  {:else}
    <a
      class="plain ws shut"
      href={switchHref(name, route.view, liveQuery.now)}
      title={name}
      style:--ws-dot={wsColor(name)}
    >
      <ChevronRight size={13} class="chev" />
      <span class="dot"></span>
      {@render wsName(name, prefix)}
      {#if workspace.usage[name]}
        <span class="num muted size">
          {formatBytes(workspace.usage[name].bytes)}
        </span>
      {/if}
    </a>
  {/if}
{/snippet}

<aside class:collapsed {inert} aria-label="navigation">
  <div class="top">
    <a class="brand plain" href={wsHref(workspace.current, 'overview')}>dw</a>
    <button
      class="bare icon"
      onclick={(e) => {
        e.currentTarget.focus() // re-expanding leaves the keyboard here
        onToggle()
      }}
      title={collapsed ? 'expand sidebar' : 'collapse sidebar'}
      aria-label={collapsed ? 'expand sidebar' : 'collapse sidebar'}
    >
      {#if collapsed}<PanelLeftOpen size={15} />{:else}<PanelLeftClose
          size={15}
        />{/if}
    </button>
  </div>

  <nav>
    <div class="group" role="group" aria-label="workspaces">
      {#if !collapsed}<span class="grouplabel muted">Workspaces</span>{/if}
      {#each folded.entries as entry (entry.kind === 'ws' ? entry.name : `group:${entry.prefix}`)}
        {#if entry.kind === 'ws'}
          {@render wsRow(entry.name)}
        {:else}
          {@const isOpen =
            openGroups.includes(entry.prefix) ||
            entry.names.includes(workspace.current)}
          {#if collapsed}
            {#each entry.names as name (name)}{@render wsRow(name)}{/each}
          {:else}
            <button
              class="plain ws shut wsgroup"
              aria-expanded={isOpen}
              onclick={() => toggleGroup(entry.prefix)}
              title="{entry.names.length} workspaces named {entry.prefix}-…"
            >
              {#if isOpen}<ChevronDown
                  size={13}
                  class="chev"
                />{:else}<ChevronRight size={13} class="chev" />{/if}
              <span class="name">{entry.prefix}-…</span>
              <span class="num muted size">{entry.names.length}</span>
            </button>
            {#if isOpen}
              <div class="groupkids">
                {#each entry.names as name (name)}
                  {@render wsRow(name, entry.prefix)}
                {/each}
              </div>
            {/if}
          {/if}
        {/if}
      {/each}
      {#if folded.hidden.length && !collapsed}
        <button
          class="bare muted hiddentoggle"
          aria-expanded={showHidden}
          onclick={() => (showHidden = !showHidden)}
        >
          <Eye size={13} />{folded.hidden.length} hidden
        </button>
        {#if showHidden}
          <div class="groupkids">
            {#each folded.hidden as name (name)}{@render wsRow(name)}{/each}
          </div>
        {/if}
      {/if}
      {#if collapsed}
        <!-- the rail has no room for the form or the reason -->
      {:else if adding}
        <NewWorkspace bind:name={draft} onclose={() => (adding = false)} />
      {:else if workspace.root}
        <button
          class="bare item"
          onclick={() => {
            draft = ''
            adding = true
          }}
          title="new workspace"
          aria-label="new workspace"
        >
          <Plus size={15} /><span class="label">new</span>
        </button>
      {:else if workspace.listed}
        <span
          class="muted noroot"
          title="Start the server with --workspace <folder> to add workspaces."
        >
          New workspaces need a workspace folder
        </span>
      {/if}
    </div>

    <div class="rule" aria-hidden="true"></div>
    <div class="group" role="group" aria-label="shared">
      {#if !collapsed}<span class="grouplabel muted">
          {SECTION_LABELS.groups.shared}
        </span>{/if}
      {#each SHARED_ITEMS as item (item.section)}
        <a
          class="plain item"
          href={sharedHref(item.section)}
          aria-current={sharedActive(item.section) ? 'page' : undefined}
          title={item.label}
        >
          <item.icon size={15} />{#if !collapsed}<span class="label">
              {item.label}
            </span>{/if}
        </a>
      {/each}
    </div>

    <div class="rule" aria-hidden="true"></div>
    <div class="group" role="group" aria-label="server">
      {#if !collapsed}<span class="grouplabel muted">
          {SECTION_LABELS.groups.server}
        </span>{/if}
      {#each SERVER_ITEMS as item (item.section)}
        <a
          class="plain item"
          href={serverHref(item.section)}
          aria-current={serverSection === item.section ? 'page' : undefined}
          title={item.label}
        >
          <item.icon size={15} />{#if !collapsed}<span class="label">
              {item.label}
            </span>{/if}
        </a>
      {/each}
    </div>
  </nav>
</aside>

<style>
  aside {
    display: flex;
    flex-direction: column;
    width: 236px;
    min-width: 236px;
    border-right: 1px solid var(--line);
    background: var(--panel);
    font-family: var(--font-mono);
    font-size: var(--t-sm);
  }
  aside.collapsed {
    width: 44px;
    min-width: 44px;
  }
  .top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0.6rem;
    border-bottom: 1px solid var(--line);
  }
  aside.collapsed .top {
    justify-content: center;
  }
  aside.collapsed .brand {
    display: none;
  }
  .brand {
    font-weight: 600;
    color: var(--ink);
  }
  .icon {
    display: inline-flex;
    padding: 0.25rem 0.45rem;
  }
  nav {
    display: flex;
    flex-direction: column;
    padding: var(--space-2) 0;
    overflow-y: auto;
  }
  .group {
    display: flex;
    flex-direction: column;
  }
  .grouplabel {
    padding: var(--space-2) var(--space-4) var(--space-1);
    font-size: var(--t-xs);
  }
  .rule {
    border-top: 1px solid var(--line);
    margin: var(--space-2) 0;
  }
  .ws.shut,
  .wsname {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: 0.35rem var(--space-3) 0.35rem var(--space-2);
    min-width: 0;
  }
  .name {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .size {
    flex: none;
    font-size: var(--t-xs);
  }
  .ws :global(.chev) {
    flex: none;
    color: var(--muted);
  }
  /* identity, not state: the workspace's own hue, the same on every page */
  .dot {
    flex: none;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--ws-dot);
  }
  .dot.solo {
    align-self: center;
    margin: var(--space-2) 0;
  }
  .ws.shut {
    color: var(--muted);
    border-left: 3px solid transparent;
  }
  .ws.shut:hover {
    color: var(--ink);
    background: var(--panel-2);
  }
  /* The user's place takes --select: a rail down the whole open block and
     a faint tint behind it */
  .ws.open {
    display: flex;
    flex-direction: column;
    margin: var(--space-1) 0;
    border-left: 3px solid var(--select);
    background: color-mix(in srgb, var(--select) 7%, transparent);
  }
  .ws.open .wsname {
    font-weight: 600;
    color: var(--ink);
  }
  .ws.open .wsname :global(.chev) {
    color: var(--select);
  }
  .ws.open.flash .wsname {
    animation: dw-settle 400ms ease-out;
  }
  @keyframes dw-settle {
    from {
      background: color-mix(in srgb, var(--select) 25%, transparent);
    }
    to {
      background: transparent;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .ws.open.flash .wsname {
      animation: none;
    }
  }
  /* the guide line that makes the sections the workspace's children */
  .sections {
    display: flex;
    flex-direction: column;
    margin: 0 0 var(--space-1) calc(var(--space-2) + 6px);
    border-left: 1px solid var(--line);
  }
  aside.collapsed .sections {
    margin: 0 0 var(--space-1);
    border-left: 0;
  }
  .item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem var(--space-4) 0.35rem calc(var(--space-4) + 0.6rem);
    color: var(--muted);
    border-left: 2px solid transparent;
    border-radius: 0;
    font: inherit;
    width: 100%;
    text-align: left;
  }
  .ws.open .item {
    padding-left: var(--space-3);
    margin-left: -1px;
  }
  aside.collapsed .item {
    padding: 0.45rem 0;
    margin-left: 0;
    justify-content: center;
  }
  .item:hover {
    color: var(--ink);
    background: var(--panel-2);
  }
  .item[aria-current='page'] {
    color: var(--ink);
    border-left-color: var(--select);
    background: color-mix(in srgb, var(--select) 16%, transparent);
    font-weight: 600;
  }
  .groupkids {
    padding-left: var(--space-3);
  }
  /* hidden rows carry the "hidden" toggle's label, not a faded colour: an
     opacity on --muted text falls below AA in both themes */
  .hiddentoggle {
    display: inline-flex;
    gap: 0.35rem;
    align-items: center;
    font-size: var(--t-sm);
    padding: var(--space-1) var(--space-2);
  }
  .wsgroup {
    width: 100%;
    text-align: left;
  }
  .dotlink {
    display: flex;
    justify-content: center;
  }
  .dotlink:hover {
    background: var(--panel-2);
  }
  .noroot {
    padding: 0.35rem var(--space-4);
    font-family: var(--font-sans);
    font-size: var(--t-xs);
  }
</style>
