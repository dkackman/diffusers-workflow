<script lang="ts">
  import { Images, Layers, ListTodo, Moon, MonitorCog, SquarePen, Sun } from 'lucide-svelte'
  import { route } from './lib/router.svelte'
  import { api } from './lib/api'
  import type { MemoryInfo } from './lib/types'
  import WorkflowsPage from './lib/pages/WorkflowsPage.svelte'
  import WorkflowPage from './lib/pages/WorkflowPage.svelte'
  import JobsPage from './lib/pages/JobsPage.svelte'
  import JobPage from './lib/pages/JobPage.svelte'
  import EditorPage from './lib/pages/EditorPage.svelte'
  import GalleryPage from './lib/pages/GalleryPage.svelte'

  let memory = $state<MemoryInfo | null>(null)
  let currentJob = $state<string | null>(null)

  $effect(() => {
    const poll = async () => {
      try {
        memory = await api.memory()
        currentJob = (await api.health()).current_job
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
    return Math.min(100, (100 * (info.gpu_memory_allocated_mb ?? 0)) / info.gpu_memory_total_mb)
  })
</script>

<header>
  <span class="brand">diffusers<span class="accent">-workflow</span></span>
  <nav>
    <a href="#/workflows" class:active={route.parts[0] === 'workflows'}>
      <Layers size={15} />Workflows
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
  </nav>
  {#if currentJob}
    <a class="runningnow" href={'#/jobs/' + currentJob} title="a job is running - click to watch">
      <span class="pulse-dot"></span>running
    </a>
  {/if}
  <span class="vram muted">
    {#if memory?.info?.gpu_available}
      {memory.info.gpu_device_name} · {gb(memory.info.gpu_memory_allocated_mb ?? 0)} /
      {gb(memory.info.gpu_memory_total_mb ?? 0)} GB
    {:else}
      worker idle
    {/if}
  </span>
  <button
    class="quiet icon themebtn"
    onclick={cycleTheme}
    title="theme: {theme} - click to change"
    aria-label="theme: {theme} - click to change"
  >
    {#if theme === 'light'}<Sun size={15} />{:else if theme === 'dark'}<Moon size={15} />{:else}<MonitorCog size={15} />{/if}
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

<main>
  {#if route.parts[0] === 'gallery'}
    <GalleryPage />
  {:else if route.parts[0] === 'edit'}
    <EditorPage name={route.parts.slice(1).join('/')} />
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
    display: flex; align-items: center; gap: 1.5rem;
    padding: 0.7rem 1.2rem; border-bottom: 1px solid var(--line);
    background: var(--panel);
    position: sticky; top: 0; z-index: 10;
  }
  .brand { font-weight: 700; font-size: 1rem; }
  .accent { color: var(--accent); }
  nav { display: flex; gap: 1.1rem; flex: 1; }
  nav a {
    display: inline-flex; align-items: center; gap: 0.35rem;
    color: var(--muted); font-weight: 600; padding: 0.2rem 0;
    border-bottom: 2px solid transparent;
  }
  nav a:hover { color: var(--ink); }
  nav a.active { color: var(--accent); border-bottom-color: var(--accent); }
  .runningnow {
    display: inline-flex; align-items: center; gap: 0.45rem;
    color: var(--accent); font-weight: 600; font-size: 0.85rem;
  }
  .vram { font-size: 0.8rem; }
  .themebtn { display: inline-flex; padding: 0.3rem 0.45rem; }
  .meter {
    width: 90px; height: 6px; border-radius: 3px;
    background: var(--panel-2); overflow: hidden;
  }
  .meter .fill { height: 100%; background: var(--accent); transition: width 0.4s ease; }
  .meter .fill.hot { background: var(--warn); }
  .meter .fill.critical { background: var(--bad); }
  main { max-width: 1100px; margin: 0 auto; padding: 1.2rem; }
</style>
