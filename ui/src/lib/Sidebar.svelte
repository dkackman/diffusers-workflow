<script lang="ts">
  import { onMount } from 'svelte'
  import {
    BookCopy,
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

  let { collapsed, onToggle }: { collapsed: boolean; onToggle: () => void } =
    $props()

  onMount(loadWorkspaces)

  // The workspace whose sections are open: the route's when it names one,
  // else the one the UI is scoped to (a shared or server page still sits
  // inside a workspace's context)
  const expanded = $derived(
    route.view.kind === 'ws' ? route.view.workspace : workspace.current,
  )
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

<aside class:collapsed aria-label="navigation">
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
    <div class="group" aria-label="workspaces">
      {#each workspace.names ?? [workspace.current] as name (name)}
        {#if name === expanded}
          <div class="ws open">
            <span class="wsname" title={name}>
              {#if !collapsed}{name}{/if}
              {#if !collapsed && workspace.usage[name]}
                <span
                  class="num muted size"
                  title="{workspace.usage[name].files} files"
                  >{formatBytes(workspace.usage[name].bytes)}</span
                >
              {/if}
            </span>
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
        {:else if !collapsed}
          <a class="plain ws shut" href={wsHref(name, 'overview')} title={name}>
            <span class="wsname">{name}</span>
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
    <div class="group" aria-label="shared">
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
    <div class="group" aria-label="server">
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
    width: 200px;
    min-width: 200px;
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
    justify-content: space-between;
    gap: var(--space-2);
    padding: 0.35rem var(--space-4);
  }
  .ws.shut {
    color: var(--muted);
  }
  .ws.shut:hover {
    color: var(--ink);
    background: var(--panel-2);
  }
  .ws.open .wsname {
    font-weight: 600;
    color: var(--ink);
  }
  .size {
    font-size: var(--t-xs);
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
  aside.collapsed .item {
    padding: 0.45rem 0;
    justify-content: center;
  }
  .item:hover {
    color: var(--ink);
    background: var(--panel-2);
  }
  /* Selection is the user's state, not the machine's: a heavier ink edge,
     never a colour */
  .item[aria-current='page'] {
    color: var(--ink);
    border-left-color: var(--ink);
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
