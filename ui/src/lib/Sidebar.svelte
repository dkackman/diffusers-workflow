<script lang="ts">
  import { onMount, untrack } from 'svelte'
  import {
    BookCopy,
    ChevronDown,
    ChevronRight,
    Database,
    FolderOpen,
    Images,
    LayoutDashboard,
    Layers,
    ListTodo,
    ListTree,
    MessageSquareText,
    PanelLeftClose,
    PanelLeftOpen,
    Plus,
    Server,
    SquarePen,
  } from '@lucide/svelte'
  import { slide } from 'svelte/transition'
  import { prefersReducedMotion } from 'svelte/motion'
  import { route } from './router.svelte'
  import {
    serverHref,
    sharedHref,
    wsHref,
    type ServerSection,
    type SharedSection,
    type WsSection,
  } from './routes'
  import { loadWorkspaces, workspace } from './workspace.svelte'
  import { createWorkspaceAndGo } from './workspaceActions'
  import { formatBytes } from './format'
  import { wsColor } from './wsColor'

  let {
    collapsed,
    onToggle,
    inert = false,
  }: { collapsed: boolean; onToggle: () => void; inert?: boolean } = $props()

  onMount(loadWorkspaces)

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

  const WS_ITEMS: { section: WsSection; label: string; icon: typeof Layers }[] =
    [
      { section: 'overview', label: 'Overview', icon: LayoutDashboard },
      { section: 'workflows', label: 'Workflows', icon: Layers },
      { section: 'jobs', label: 'Jobs', icon: ListTodo },
      { section: 'gallery', label: 'Gallery', icon: Images },
      { section: 'assets', label: 'Assets', icon: FolderOpen },
      { section: 'edit', label: 'Editor', icon: SquarePen },
    ]
  const SHARED_ITEMS: {
    section: SharedSection
    label: string
    icon: typeof Layers
  }[] = [
    { section: 'prompts', label: 'Prompts', icon: MessageSquareText },
    { section: 'assets', label: 'Assets', icon: FolderOpen },
    { section: 'examples', label: 'Examples', icon: BookCopy },
  ]
  const SERVER_ITEMS: {
    section: ServerSection
    label: string
    icon: typeof Layers
  }[] = [
    { section: 'models', label: 'Models', icon: Database },
    { section: 'schema', label: 'Schema', icon: ListTree },
    { section: 'status', label: 'Status', icon: Server },
  ]
  // prompt-edit sits under Prompts in the nav
  const sharedActive = (s: SharedSection) =>
    sharedSection === s || (s === 'prompts' && sharedSection === 'prompt-edit')

  let adding = $state(false)
  let newName = $state('')
  let addError = $state('')
  let nameInput = $state<HTMLInputElement | null>(null)

  function startAdd() {
    adding = true
    addError = ''
    newName = ''
    // the input mounts on the next tick
    queueMicrotask(() => nameInput?.focus())
  }
  async function submitAdd() {
    const failure = await createWorkspaceAndGo(newName.trim())
    if (failure) addError = failure
    else adding = false
  }
</script>

<aside class:collapsed {inert} aria-label="navigation">
  <div class="top">
    <a class="brand plain" href={wsHref(workspace.current, 'overview')}>dw</a>
    <button
      class="bare icon"
      onclick={onToggle}
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
      {#each workspace.names ?? [workspace.current] as name (name)}
        {#if name === expanded}
          <!-- The open workspace is one block: a --select rail and tint
               behind its name and sections, a guide line down its children,
               so siblings read as outside it -->
          <div
            class="ws open"
            class:flash={switched}
            style:--ws-dot={wsColor(name)}
          >
            {#if collapsed}
              <span class="dot solo" title={name}></span>
            {:else}
              <span class="wsname" title={name}>
                <ChevronDown size={13} class="chev" />
                <span class="dot"></span>
                <span class="name">{name}</span>
                {#if workspace.usage[name]}
                  <span
                    class="num muted size"
                    title="{workspace.usage[name].files} files"
                    >{formatBytes(workspace.usage[name].bytes)}</span
                  >
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
                  <item.icon size={15} />{#if !collapsed}<span class="label"
                      >{item.label}</span
                    >{/if}
                </a>
              {/each}
            </div>
          </div>
        {:else if !collapsed}
          <a
            class="plain ws shut"
            href={wsHref(name, 'overview')}
            title={name}
            style:--ws-dot={wsColor(name)}
          >
            <ChevronRight size={13} class="chev" />
            <span class="dot"></span>
            <span class="name">{name}</span>
            {#if workspace.usage[name]}
              <span class="num muted size"
                >{formatBytes(workspace.usage[name].bytes)}</span
              >
            {/if}
          </a>
        {/if}
      {/each}
      {#if !collapsed && workspace.root}
        {#if adding}
          <form
            class="newws"
            onsubmit={(e) => {
              e.preventDefault()
              submitAdd()
            }}
          >
            <input
              bind:this={nameInput}
              bind:value={newName}
              placeholder="name"
              aria-label="new workspace name"
              onkeydown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  submitAdd()
                }
                if (e.key === 'Escape') adding = false
              }}
            />
            {#if addError}<span class="muted err">{addError}</span>{/if}
          </form>
        {:else}
          <button
            class="bare item"
            onclick={startAdd}
            title="new workspace"
            aria-label="new workspace"
          >
            <Plus size={15} /><span class="label">new</span>
          </button>
        {/if}
      {/if}
    </div>

    <div class="rule" aria-hidden="true"></div>
    <div class="group" role="group" aria-label="shared">
      {#if !collapsed}<span class="grouplabel muted">Shared</span>{/if}
      {#each SHARED_ITEMS as item (item.section)}
        <a
          class="plain item"
          href={sharedHref(item.section)}
          aria-current={sharedActive(item.section) ? 'page' : undefined}
          title={item.label}
        >
          <item.icon size={15} />{#if !collapsed}<span class="label"
              >{item.label}</span
            >{/if}
        </a>
      {/each}
    </div>

    <div class="rule" aria-hidden="true"></div>
    <div class="group" role="group" aria-label="server">
      {#if !collapsed}<span class="grouplabel muted">Server</span>{/if}
      {#each SERVER_ITEMS as item (item.section)}
        <a
          class="plain item"
          href={serverHref(item.section)}
          aria-current={serverSection === item.section ? 'page' : undefined}
          title={item.label}
        >
          <item.icon size={15} />{#if !collapsed}<span class="label"
              >{item.label}</span
            >{/if}
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
  .newws {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    padding: 0.2rem var(--space-4);
  }
  .newws input {
    width: 100%;
    font-family: var(--font-mono);
    font-size: var(--t-sm);
  }
  .err {
    font-family: var(--font-sans);
    font-size: var(--t-xs);
  }
</style>
