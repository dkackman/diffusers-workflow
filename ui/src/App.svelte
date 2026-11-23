<script lang="ts">
  import { startStatusPoll } from './lib/statusPoll'
  import { connection, quietWord } from './lib/connection.svelte'
  import ConnectionBanner from './lib/ui/ConnectionBanner.svelte'
  import { cardReadings, meterText, meterTitle } from './lib/cardMemory'
  import {
    BookOpen,
    Braces,
    ChevronDown,
    KeyRound,
    Menu,
    Moon,
    MonitorCog,
    Sun,
  } from '@lucide/svelte'
  import { Toaster } from 'svelte-sonner'
  import { tick, untrack } from 'svelte'
  import { route } from './lib/router.svelte'
  import { isEditable } from './lib/editable'
  import { runningIds } from './lib/jobWatch'
  import { jobHref } from './lib/routes'
  import {
    alertMode,
    clearEndedMark,
    followJobs,
    unlockChime,
  } from './lib/finishAlert'
  import { alerts } from './lib/alerts.svelte'
  import { cycleTheme, initialTheme, type Theme } from './lib/theme'
  import { storageGet, storageSet } from './lib/storage'
  import Sidebar from './lib/Sidebar.svelte'
  import Breadcrumb from './lib/Breadcrumb.svelte'
  import KeyboardHelp from './lib/KeyboardHelp.svelte'
  import StatusPopover from './lib/StatusPopover.svelte'
  import TokenPopover from './lib/TokenPopover.svelte'
  import AlertToggle from './lib/AlertToggle.svelte'
  import ConfirmDialog from './lib/ui/ConfirmDialog.svelte'
  import WorkflowsPage from './lib/pages/WorkflowsPage.svelte'
  import WorkflowPage from './lib/pages/WorkflowPage.svelte'
  import NotFoundPage from './lib/pages/NotFoundPage.svelte'
  import JobsPage from './lib/pages/JobsPage.svelte'
  import JobPage from './lib/pages/JobPage.svelte'
  import RequestsPage from './lib/pages/RequestsPage.svelte'
  import RequestPage from './lib/pages/RequestPage.svelte'
  import EditorPage from './lib/pages/EditorPage.svelte'
  import PromptsPage from './lib/pages/PromptsPage.svelte'
  import PromptEditorPage from './lib/pages/PromptEditorPage.svelte'
  import GalleryPage from './lib/pages/GalleryPage.svelte'
  import ComparePage from './lib/pages/ComparePage.svelte'
  import OverviewPage from './lib/pages/OverviewPage.svelte'
  import AssetsPage from './lib/pages/AssetsPage.svelte'
  import ModelsPage from './lib/pages/ModelsPage.svelte'
  import SchemaPage from './lib/pages/SchemaPage.svelte'
  import StatusPage from './lib/pages/StatusPage.svelte'

  const health = $derived(connection.health)
  let helpOpen = $state(false)
  let statusOpen = $state(false)
  let tokenOpen = $state(false)
  // The popovers sit against their triggers, and leave a press on them to
  // the trigger's own toggle
  let statusTrigger = $state<HTMLElement | null>(null)
  let tokenTrigger = $state<HTMLElement | null>(null)
  const currentJob = $derived(health?.current_job ?? null)
  // One job per card: with more than one running, the chip cannot link to
  // them all, so it joins the status trigger and the popover lists each
  const running = $derived(health ? runningIds(health) : [])
  const statusLabel = $derived(
    running.length > 1
      ? `${running.length} jobs running - server & worker status`
      : 'server & worker status',
  )
  const view = $derived(route.view)

  // The tab title and the finish alert (finishAlert.ts)
  const onItsPage = (id: string) =>
    view.kind === 'ws' && view.section === 'jobs' && view.rest[0] === id

  $effect(() => {
    const back = clearEndedMark
    // A chime may only play after a gesture; after a reload the first
    // click anywhere is that gesture
    const unlock = () => {
      if (alerts.on && alertMode() === 'chime') unlockChime()
    }
    document.addEventListener('visibilitychange', back)
    window.addEventListener('pointerdown', unlock, { once: true })
    return () => {
      document.removeEventListener('visibilitychange', back)
      window.removeEventListener('pointerdown', unlock)
    }
  })

  function onKeydown(event: KeyboardEvent) {
    if (event.key === '?' && !isEditable(event.target)) {
      event.preventDefault()
      helpOpen = true
    } else if (event.key === 'Escape' && drawerOpen) {
      event.preventDefault()
      drawerOpen = false
    }
  }

  // The one timer that keeps going while the tab is hidden (statusPoll.ts);
  // each reading it lands moves the watched jobs on
  $effect(() => startStatusPoll())
  $effect(() => {
    const info = health
    untrack(
      () => void followJobs(info, { onItsPage, alertsOn: () => alerts.on }),
    )
  })

  let theme = $state<Theme>(initialTheme())
  // one meter bar per card, from the per-card readings; the top level of
  // /api/memory is only the first card's
  const metered = $derived(
    cardReadings(connection.memory).filter(
      (c) => c.available && c.pct !== null,
    ),
  )

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
  // Focus follows the drawer: in to its first link on open, back to the
  // menu button on close (the scrim, Escape or a navigation)
  let menuButton = $state<HTMLElement | null>(null)
  let navShell = $state<HTMLElement | null>(null)
  let drawerWas = false
  $effect(() => {
    const open = drawerOpen
    if (open === drawerWas) return
    drawerWas = open
    void tick().then(() => {
      if (!narrow) return
      if (open) navShell?.querySelector<HTMLElement>('aside a')?.focus()
      else menuButton?.focus()
    })
  })
  function skipToContent(event: Event) {
    // the router reads the hash, so the link must not navigate
    event.preventDefault()
    document.getElementById('main')?.focus()
  }
  $effect(() => {
    // any navigation closes the drawer and scrolls back to the top
    void route.parts
    drawerOpen = false
    window.scrollTo(0, 0)
  })
  const wide = $derived(
    (view.kind === 'ws' && view.section === 'edit') ||
      (view.kind === 'shared' && view.section === 'prompt-edit'),
  )
</script>

<svelte:window onkeydown={onKeydown} />

<!-- The sidebar carries navigation; the header carries where you are and what the GPU is doing -->
<div class="shell" class:drawer={drawerOpen} bind:this={navShell}>
  <a class="skiplink" href="#main" onclick={skipToContent}>Skip to content</a>
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
          bind:this={menuButton}
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
          {#if running.length === 1}
            <!-- The one thing worth pinning to every page: what the GPU is
             doing, and a way straight to it. The legacy '#/jobs/<id>' form is
             deliberate: health carries no workspace for the job, so the legacy
             redirect plus JobPage's correction is how it lands under the job's
             own workspace -->
            <a
              class="plain live"
              href={jobHref(running[0])}
              title="go to the job"
            >
              <span class="pulse-dot"></span>running
            </a>
          {/if}
          {#snippet busy()}
            <!-- Inside the trigger rather than beside it: a second button
             opening the same popover would read as a press outside it -->
            <span class="live">
              <span class="pulse-dot"></span>{running.length} running
              <ChevronDown size={13} />
            </span>
          {/snippet}
          {#if metered.length}
            <button
              class="bare vram"
              bind:this={statusTrigger}
              onclick={() => (statusOpen = !statusOpen)}
              title={meterTitle(metered) || 'VRAM allocated'}
              aria-label="{statusLabel}, {meterText(metered)}"
              aria-expanded={statusOpen}
            >
              {#if running.length > 1}{@render busy()}{:else}<span
                  class="stateword"
                >
                  {running.length ? 'running' : 'idle'}
                </span>{/if}
              <span class="meters">
                {#each metered as card, i (card.device ?? i)}
                  <span class="meter">
                    <span
                      class="fill"
                      class:hot={card.pct! > 75}
                      class:critical={card.pct! > 92}
                      style:width={card.pct + '%'}
                    ></span>
                  </span>
                {/each}
              </span>
              <span class="num vramtext">{meterText(metered)}</span>
            </button>
          {:else}
            <button
              class="bare"
              class:muted={currentJob === null}
              bind:this={statusTrigger}
              onclick={() => (statusOpen = !statusOpen)}
              title="server & worker status"
              aria-label={statusLabel}
              aria-expanded={statusOpen}
            >
              {#if running.length > 1}{@render busy()}
              {:else}{currentJob ? 'status' : quietWord(connection.state)}{/if}
            </button>
          {/if}
          <StatusPopover bind:open={statusOpen} anchor={statusTrigger} />
          <button
            class="bare icon"
            bind:this={tokenTrigger}
            onclick={() => (tokenOpen = !tokenOpen)}
            title="API token"
            aria-label="API token"
            aria-expanded={tokenOpen}
          >
            <KeyRound size={15} />
          </button>
          <TokenPopover bind:open={tokenOpen} anchor={tokenTrigger} />
          <AlertToggle />
          <button
            class="bare icon"
            onclick={() => (theme = cycleTheme(theme))}
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
    <ConnectionBanner ontoken={() => (tokenOpen = true)} />

    <main id="main" tabindex="-1" class:wide>
      {#if view.kind === 'missing'}<NotFoundPage />
      {:else if view.kind === 'server'}
        {#if view.section === 'schema'}<SchemaPage />
        {:else if view.section === 'models'}<ModelsPage />
        {:else}<StatusPage />{/if}
      {:else if view.kind === 'shared'}
        {#if view.section === 'prompts'}<PromptsPage />
        {:else if view.section === 'prompt-edit'}<PromptEditorPage
            name={view.rest.join('/')}
          />
        {:else if view.section === 'assets'}<AssetsPage shared />
        {:else if view.rest.length}<WorkflowPage name={view.rest.join('/')} />
        {:else}<WorkflowsPage examples />{/if}
      {:else if view.section === 'overview'}<OverviewPage />
      {:else if view.section === 'gallery'}<GalleryPage />
      {:else if view.section === 'compare'}<ComparePage names={view.rest} />
      {:else if view.section === 'assets'}<AssetsPage />
      {:else if view.section === 'edit'}<EditorPage
          name={view.rest.join('/')}
        />
      {:else if view.section === 'jobs' && view.rest[0]}<JobPage
          jobId={view.rest[0]}
        />
      {:else if view.section === 'jobs'}<JobsPage />
      {:else if view.section === 'requests' && view.rest[0]}<RequestPage
          requestId={view.rest[0]}
        />
      {:else if view.section === 'requests'}<RequestsPage />
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
    /* header and main are flex items of .column: without this, a flex
       item's automatic minimum width is its content's min-content size
       (a wide table, a crowded icon row), which grows the item - and the
       document - past the viewport instead of letting the item's own
       overflow handling (a scroller, wrapping) take over */
    min-width: 0;
  }
  .navrow {
    display: flex;
    align-items: center;
    gap: 0.5rem 1rem;
    padding: 0.55rem 1.2rem;
  }
  .skiplink {
    position: absolute;
    left: 0.5rem;
    top: 0.5rem;
    z-index: 30;
    padding: 0.4rem 0.7rem;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-1);
    transform: translateY(-200%);
  }
  .skiplink:focus {
    transform: none;
  }
  main:focus {
    outline: none;
  }
  .stateword {
    font-size: var(--t-xs);
    color: var(--muted);
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
  .meters {
    display: flex;
    flex-direction: column;
    gap: 2px;
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
    /* main is a flex item of .column (flex-direction: column); auto
       cross-axis margins (the `margin: 0 auto` that centers it) opt it
       out of the default stretch sizing, so without an explicit width it
       falls back to fit-content and grows to whatever its widest
       descendant (a wide table, a crowded header row) wants - taking the
       document past the viewport instead of leaving that descendant's own
       overflow handling (a scroller, wrapping) to cope */
    width: 100%;
    max-width: 1180px;
    margin: 0 auto;
    padding: 1.6rem 1.2rem 4rem;
    min-width: 0;
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
      /* the menu button, breadcrumb and state icons no longer fit one row
         on a phone - wrap rather than overflow the document. Harmless
         above that width since flex-wrap only engages when the row
         genuinely does not fit */
      flex-wrap: wrap;
    }
    .state {
      margin-left: auto;
    }
    main {
      padding: 1rem 0.8rem 3rem;
    }
    .meter {
      display: none;
    }
  }
</style>
