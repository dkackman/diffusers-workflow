<script lang="ts">
  import {
    BookOpen,
    Braces,
    KeyRound,
    Menu,
    Moon,
    MonitorCog,
    Sun,
  } from '@lucide/svelte'
  import { Toaster } from 'svelte-sonner'
  import { route } from './lib/router.svelte'
  import { api } from './lib/api'
  import { storageGet, storageSet } from './lib/storage'
  import type { HealthInfo, MemoryInfo } from './lib/types'
  import Sidebar from './lib/Sidebar.svelte'
  import Breadcrumb from './lib/Breadcrumb.svelte'
  import KeyboardHelp from './lib/KeyboardHelp.svelte'
  import StatusPopover from './lib/StatusPopover.svelte'
  import TokenPopover from './lib/TokenPopover.svelte'
  import ConfirmDialog from './lib/ConfirmDialog.svelte'
  import WorkflowsPage from './lib/pages/WorkflowsPage.svelte'
  import WorkflowPage from './lib/pages/WorkflowPage.svelte'
  import JobsPage from './lib/pages/JobsPage.svelte'
  import JobPage from './lib/pages/JobPage.svelte'
  import EditorPage from './lib/pages/EditorPage.svelte'
  import PromptsPage from './lib/pages/PromptsPage.svelte'
  import PromptEditorPage from './lib/pages/PromptEditorPage.svelte'
  import GalleryPage from './lib/pages/GalleryPage.svelte'
  import AssetsPage from './lib/pages/AssetsPage.svelte'
  import ModelsPage from './lib/pages/ModelsPage.svelte'
  import SchemaPage from './lib/pages/SchemaPage.svelte'
  import ServerPage from './lib/pages/ServerPage.svelte'

  let memory = $state<MemoryInfo | null>(null)
  let health = $state<HealthInfo | null>(null)
  let helpOpen = $state(false)
  let statusOpen = $state(false)
  let tokenOpen = $state(false)
  const currentJob = $derived(health?.current_job ?? null)

  function isEditable(target: EventTarget | null): boolean {
    if (!(target instanceof HTMLElement)) return false
    return (
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLSelectElement ||
      target.isContentEditable
    )
  }

  function onKeydown(event: KeyboardEvent) {
    if (event.key === '?' && !isEditable(event.target)) {
      event.preventDefault()
      helpOpen = true
    } else if (event.key === 'Escape' && helpOpen) {
      event.preventDefault()
      helpOpen = false
    } else if (event.key === 'Escape' && statusOpen) {
      event.preventDefault()
      statusOpen = false
    } else if (event.key === 'Escape' && tokenOpen) {
      event.preventDefault()
      tokenOpen = false
    } else if (event.key === 'Escape' && drawerOpen) {
      event.preventDefault()
      drawerOpen = false
    }
  }

  $effect(() => {
    const poll = async () => {
      // Settled independently: memory answers 503 while the worker is
      // unreachable, and that must not blank the "running" indicator too
      const [memoryInfo, healthInfo] = await Promise.allSettled([
        api.memory(),
        api.health(),
      ])
      memory = memoryInfo.status === 'fulfilled' ? memoryInfo.value : null
      health = healthInfo.status === 'fulfilled' ? healthInfo.value : null
    }
    poll()
    const timer = setInterval(poll, 5000)
    return () => clearInterval(timer)
  })

  const gb = (mb: number) => (mb / 1024).toFixed(1)

  type Theme = 'system' | 'light' | 'dark'
  let theme = $state<Theme>(
    (() => {
      try {
        const stored = localStorage.getItem('dw-theme')
        return stored === 'light' || stored === 'dark' ? stored : 'system'
      } catch {
        return 'system'
      }
    })(),
  )

  function cycleTheme() {
    theme = theme === 'system' ? 'light' : theme === 'light' ? 'dark' : 'system'
    if (theme === 'system') delete document.documentElement.dataset.theme
    else document.documentElement.dataset.theme = theme
    try {
      if (theme === 'system') localStorage.removeItem('dw-theme')
      else localStorage.setItem('dw-theme', theme)
    } catch {
      /* fine - applies for this session */
    }
  }
  const vramPct = $derived.by(() => {
    const info = memory?.info
    if (!info?.gpu_available || !info.gpu_memory_total_mb) return null
    return Math.min(
      100,
      (100 * (info.gpu_memory_allocated_mb ?? 0)) / info.gpu_memory_total_mb,
    )
  })

  // storage.ts namespaces the key: this is localStorage 'dw-sidebar'
  let sidebarCollapsed = $state(storageGet<boolean>('sidebar', false))
  function toggleSidebar() {
    sidebarCollapsed = !sidebarCollapsed
    storageSet('sidebar', sidebarCollapsed)
  }
  // Narrow viewports: the sidebar is an overlay opened from the header.
  // While it is off screen it is inert, so a keyboard or screen reader
  // does not walk its links before reaching the menu button
  let drawerOpen = $state(false)
  let narrow = $state(false)
  $effect(() => {
    const query = window.matchMedia('(max-width: 900px)')
    const apply = () => (narrow = query.matches)
    apply()
    query.addEventListener('change', apply)
    return () => query.removeEventListener('change', apply)
  })
  $effect(() => {
    // any navigation closes the drawer
    void route.parts
    drawerOpen = false
  })
  const view = $derived(route.view)
  const wide = $derived(
    (view.kind === 'ws' && view.section === 'edit') ||
      (view.kind === 'shared' && view.section === 'prompt-edit'),
  )
</script>

<svelte:window onkeydown={onKeydown} />

<!-- The sidebar carries navigation; the header carries where you are and what the GPU is doing -->
<div class="shell" class:drawer={drawerOpen}>
  <Sidebar
    collapsed={sidebarCollapsed && !drawerOpen}
    inert={narrow && !drawerOpen}
    onToggle={toggleSidebar}
  />
  {#if drawerOpen}
    <button
      class="scrim"
      aria-label="close navigation"
      onclick={() => (drawerOpen = false)}
    ></button>
  {/if}
  <div class="column">
    <header>
      <div class="navrow">
        <button
          class="bare icon menu"
          onclick={() => (drawerOpen = !drawerOpen)}
          aria-label="open navigation"
          aria-expanded={drawerOpen}
          title="navigation"
        >
          <Menu size={15} />
        </button>
        <Breadcrumb />
        <span class="flex"></span>
        <div class="state">
          {#if currentJob}
            <!-- The one thing worth pinning to every page: what the GPU is
             doing, and a way straight to it -->
            <a
              class="plain live"
              href="#/jobs/{currentJob}"
              title="go to the job"
            >
              <span class="pulse-dot"></span>running
            </a>
          {/if}
          {#if vramPct !== null}
            <button
              class="bare vram"
              onclick={(e) => {
                e.stopPropagation()
                statusOpen = !statusOpen
              }}
              title={memory?.info?.gpu_device_name
                ? `${memory.info.gpu_device_name} - ${gb(memory.info.gpu_memory_allocated_mb ?? 0)} of ${gb(memory.info.gpu_memory_total_mb ?? 0)} GB allocated`
                : 'VRAM allocated'}
              aria-label="server & worker status"
              aria-expanded={statusOpen}
            >
              <span class="meter">
                <span
                  class="fill"
                  class:hot={vramPct > 75}
                  class:critical={vramPct > 92}
                  style:width={vramPct + '%'}
                ></span>
              </span>
              <span class="num vramtext"
                >{gb(memory?.info?.gpu_memory_allocated_mb ?? 0)}/{gb(
                  memory?.info?.gpu_memory_total_mb ?? 0,
                )} GB</span
              >
            </button>
          {:else}
            <button
              class="bare"
              class:muted={currentJob === null}
              onclick={(e) => {
                e.stopPropagation()
                statusOpen = !statusOpen
              }}
              title="server & worker status"
              aria-label="server & worker status"
              aria-expanded={statusOpen}
            >
              {currentJob ? 'status' : 'idle'}
            </button>
          {/if}
          <StatusPopover bind:open={statusOpen} {health} {memory} />
          <button
            class="bare icon"
            onclick={(e) => {
              e.stopPropagation()
              tokenOpen = !tokenOpen
            }}
            title="API token"
            aria-label="API token"
            aria-expanded={tokenOpen}
          >
            <KeyRound size={15} />
          </button>
          <TokenPopover bind:open={tokenOpen} />
          <button
            class="bare icon"
            onclick={cycleTheme}
            title="theme: {theme} - click to change"
            aria-label="theme: {theme} - click to change"
          >
            {#if theme === 'light'}<Sun
                size={15}
              />{:else if theme === 'dark'}<Moon size={15} />{:else}<MonitorCog
                size={15}
              />{/if}
          </button>
          <a
            class="plain helplink"
            href="https://github.com/dkackman/diffusers-workflow#documentation"
            target="_blank"
            rel="noopener"
            title="documentation on GitHub"
            aria-label="documentation on GitHub"
          >
            <BookOpen size={15} />
          </a>
          <a
            class="plain helplink"
            href="/docs"
            target="_blank"
            rel="noopener"
            title="interactive API reference (OpenAPI)"
            aria-label="interactive API reference (OpenAPI)"
          >
            <Braces size={15} />
          </a>
        </div>
      </div>
    </header>

    <main class:wide>
      {#if view.kind === 'server'}
        {#if view.section === 'schema'}<SchemaPage />
        {:else if view.section === 'models'}<ModelsPage />
        {:else}<ServerPage />{/if}
      {:else if view.kind === 'shared'}
        {#if view.section === 'prompts'}<PromptsPage />
        {:else if view.section === 'prompt-edit'}<PromptEditorPage
            name={view.rest.join('/')}
          />
        {:else if view.section === 'assets'}<AssetsPage />
        {:else if view.rest.length}<WorkflowPage name={view.rest.join('/')} />
        {:else}<WorkflowsPage />{/if}
      {:else if view.section === 'gallery'}<GalleryPage />
      {:else if view.section === 'assets'}<AssetsPage />
      {:else if view.section === 'edit'}<EditorPage
          name={view.rest.join('/')}
        />
      {:else if view.section === 'jobs' && view.rest[0]}<JobPage
          jobId={view.rest[0]}
        />
      {:else if view.section === 'jobs'}<JobsPage />
      {:else if view.section === 'workflows' && view.rest.length}<WorkflowPage
          name={view.rest.join('/')}
        />
      {:else}<WorkflowsPage />{/if}
    </main>
  </div>
</div>

<Toaster position="bottom-right" closeButton {theme} duration={4000} />
<ConfirmDialog />

<KeyboardHelp bind:open={helpOpen} />

<style>
  .shell {
    display: flex;
    min-height: 100vh;
  }
  .column {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
  }
  header {
    border-bottom: 1px solid var(--line);
    background: var(--panel);
    position: sticky;
    top: 0;
    z-index: 10;
  }
  .navrow {
    display: flex;
    align-items: center;
    gap: 0.5rem 1rem;
    padding: 0.55rem 1.2rem;
  }
  .menu {
    display: none;
  }
  .scrim {
    display: none;
  }
  .state {
    position: relative;
    display: flex;
    align-items: center;
    gap: 0.3rem;
    font-size: var(--t-sm);
  }
  .state :global(button) {
    padding: 0.25rem 0.45rem;
  }
  .state :global(button.icon) {
    display: inline-flex;
  }
  /* The signal colour, earning its keep: the GPU is busy */
  .live {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--live);
    font-weight: 600;
    padding: 0.25rem 0.5rem;
    border-radius: var(--radius-1);
    background: color-mix(in srgb, var(--live) 12%, transparent);
  }
  .vram {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
  }
  .vramtext {
    font-size: var(--t-xs);
    white-space: nowrap;
  }
  .meter {
    display: block;
    width: 54px;
    height: 5px;
    border-radius: 3px;
    background: var(--panel-2);
    overflow: hidden;
    flex: none;
  }
  .meter .fill {
    display: block;
    height: 100%;
    background: var(--muted);
    transition: width 0.4s ease;
  }
  /* Pressure is machine state, so it takes the signal colour */
  .meter .fill.hot {
    background: var(--live);
  }
  .meter .fill.critical {
    background: var(--bad);
  }
  .helplink {
    display: inline-flex;
    align-items: center;
    color: var(--muted);
    padding: 0.25rem;
    border-radius: var(--radius-1);
  }
  .helplink:hover {
    color: var(--ink);
    background: var(--panel-2);
  }
  main {
    max-width: 1180px;
    margin: 0 auto;
    padding: 1.6rem 1.2rem 4rem;
  }
  main.wide {
    max-width: 1560px;
  }

  /* Below the breakpoint the sidebar is a drawer opened from the header
     button and closed by the scrim or any navigation */
  @media (max-width: 900px) {
    .menu {
      display: inline-flex;
      padding: 0.25rem 0.45rem;
    }
    .shell :global(aside) {
      position: fixed;
      inset: 0 auto 0 0;
      z-index: 20;
      transform: translateX(-100%);
      transition: transform 0.15s ease;
    }
    .shell.drawer :global(aside) {
      transform: none;
    }
    .shell.drawer .scrim {
      display: block;
      position: fixed;
      inset: 0;
      z-index: 15;
      background: rgb(0 0 0 / 0.35);
      border: 0;
      border-radius: 0;
      padding: 0;
    }
    .vramtext {
      display: none;
    }
  }
  @media (max-width: 640px) {
    .navrow {
      padding: 0.5rem 0.8rem;
    }
    main {
      padding: 1rem 0.8rem 3rem;
    }
    .meter {
      display: none;
    }
  }
</style>
