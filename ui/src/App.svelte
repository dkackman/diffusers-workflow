<script lang="ts">
  import {
    BookOpen,
    Braces,
    Database,
    Images,
    Layers,
    ListTodo,
    ListTree,
    MessageSquareText,
    Moon,
    MonitorCog,
    SquarePen,
    Sun,
  } from '@lucide/svelte'
  import { route } from './lib/router.svelte'
  import { api } from './lib/api'
  import type { MemoryInfo } from './lib/types'
  import WorkflowsPage from './lib/pages/WorkflowsPage.svelte'
  import WorkflowPage from './lib/pages/WorkflowPage.svelte'
  import JobsPage from './lib/pages/JobsPage.svelte'
  import JobPage from './lib/pages/JobPage.svelte'
  import EditorPage from './lib/pages/EditorPage.svelte'
  import PromptsPage from './lib/pages/PromptsPage.svelte'
  import PromptEditorPage from './lib/pages/PromptEditorPage.svelte'
  import GalleryPage from './lib/pages/GalleryPage.svelte'
  import ModelsPage from './lib/pages/ModelsPage.svelte'
  import SchemaPage from './lib/pages/SchemaPage.svelte'

  let memory = $state<MemoryInfo | null>(null)
  let currentJob = $state<string | null>(null)

  $effect(() => {
    const poll = async () => {
      try {
        const [memoryInfo, health] = await Promise.all([
          api.memory(),
          api.health(),
        ])
        memory = memoryInfo
        currentJob = health.current_job
      } catch {
        memory = null
        currentJob = null
      }
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
</script>

<header>
  <span class="brand">diffusers<span class="accent">-workflow</span></span>
  <nav>
    <a href="#/workflows" class:active={route.parts[0] === 'workflows'}>
      <Layers size={15} />Workflows
    </a>
    <a
      href="#/prompts"
      class:active={route.parts[0] === 'prompts' ||
        route.parts[0] === 'prompt-edit'}
    >
      <MessageSquareText size={15} />Prompts
    </a>
    <a href="#/jobs" class:active={route.parts[0] === 'jobs'}>
      <ListTodo size={15} />Jobs
    </a>
    <a href="#/edit" class:active={route.parts[0] === 'edit'}>
      <SquarePen size={15} />Editor
    </a>
    <a href="#/gallery" class:active={route.parts[0] === 'gallery'}>
      <Images size={15} />Gallery
    </a>
    <a href="#/models" class:active={route.parts[0] === 'models'}>
      <Database size={15} />Models
    </a>
    <a href="#/schema" class:active={route.parts[0] === 'schema'}>
      <ListTree size={15} />Schema
    </a>
  </nav>
  {#if currentJob}
    <a
      class="runningnow"
      href={'#/jobs/' + currentJob}
      title="a job is running - click to watch"
    >
      <span class="pulse-dot"></span>running
    </a>
  {/if}
  <span class="vram muted">
    {#if memory?.info?.gpu_available}
      {memory.info.gpu_device_name} · {gb(
        memory.info.gpu_memory_allocated_mb ?? 0,
      )} /
      {gb(memory.info.gpu_memory_total_mb ?? 0)} GB
    {:else}
      worker idle
    {/if}
  </span>
  <a
    class="helplink"
    href="https://github.com/dkackman/diffusers-workflow#documentation"
    target="_blank"
    rel="noopener"
    title="documentation on GitHub"
    aria-label="documentation on GitHub"
  >
    <BookOpen size={15} />
  </a>
  <a
    class="helplink"
    href="/docs"
    target="_blank"
    rel="noopener"
    title="interactive API reference (OpenAPI)"
    aria-label="interactive API reference (OpenAPI)"
  >
    <Braces size={15} />
  </a>
  <button
    class="quiet icon themebtn"
    onclick={cycleTheme}
    title="theme: {theme} - click to change"
    aria-label="theme: {theme} - click to change"
  >
    {#if theme === 'light'}<Sun size={15} />{:else if theme === 'dark'}<Moon
        size={15}
      />{:else}<MonitorCog size={15} />{/if}
  </button>
  {#if vramPct !== null}
    <div class="meter" title="VRAM allocated">
      <div
        class="fill"
        class:hot={vramPct > 75}
        class:critical={vramPct > 92}
        style:width={vramPct + '%'}
      ></div>
    </div>
  {/if}
</header>

<main
  class:wide={route.parts[0] === 'edit' || route.parts[0] === 'prompt-edit'}
>
  {#if route.parts[0] === 'schema'}
    <SchemaPage />
  {:else if route.parts[0] === 'models'}
    <ModelsPage />
  {:else if route.parts[0] === 'gallery'}
    <GalleryPage />
  {:else if route.parts[0] === 'edit'}
    <EditorPage name={route.parts.slice(1).join('/')} />
  {:else if route.parts[0] === 'prompt-edit'}
    <PromptEditorPage name={route.parts.slice(1).join('/')} />
  {:else if route.parts[0] === 'prompts'}
    <PromptsPage />
  {:else if route.parts[0] === 'jobs' && route.parts[1]}
    <JobPage jobId={route.parts[1]} />
  {:else if route.parts[0] === 'jobs'}
    <JobsPage />
  {:else if route.parts[0] === 'workflows' && route.parts[1]}
    <WorkflowPage name={route.parts.slice(1).join('/')} />
  {:else}
    <WorkflowsPage />
  {/if}
</main>

<style>
  header {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 0.7rem 1.2rem;
    border-bottom: 1px solid var(--line);
    background: var(--panel);
    position: sticky;
    top: 0;
    z-index: 10;
  }
  .brand {
    font-weight: 700;
    font-size: 1rem;
  }
  .accent {
    color: var(--accent);
  }
  nav {
    display: flex;
    gap: 1.1rem;
    flex: 1;
  }
  nav a {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    color: var(--muted);
    font-weight: 600;
    padding: 0.2rem 0;
    border-bottom: 2px solid transparent;
  }
  nav a:hover {
    color: var(--ink);
  }
  nav a.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }
  .helplink {
    display: inline-flex;
    align-items: center;
    color: var(--muted);
    padding: 0.2rem;
  }
  .helplink:hover {
    color: var(--ink);
  }
  .runningnow {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    color: var(--accent);
    font-weight: 600;
    font-size: 0.85rem;
  }
  .vram {
    font-size: 0.8rem;
  }
  .themebtn {
    display: inline-flex;
    padding: 0.3rem 0.45rem;
  }
  .meter {
    width: 90px;
    height: 6px;
    border-radius: 3px;
    background: var(--panel-2);
    overflow: hidden;
  }
  .meter .fill {
    height: 100%;
    background: var(--accent);
    transition: width 0.4s ease;
  }
  .meter .fill.hot {
    background: var(--warn);
  }
  .meter .fill.critical {
    background: var(--bad);
  }
  main {
    max-width: 1100px;
    margin: 0 auto;
    padding: 1.2rem;
  }
  main.wide {
    max-width: 1560px;
  }
</style>
