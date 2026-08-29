<script lang="ts">
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

  $effect(() => {
    const poll = async () => {
      try {
        memory = await api.memory()
      } catch {
        memory = null
      }
    }
    poll()
    const timer = setInterval(poll, 5000)
    return () => clearInterval(timer)
  })

  const gb = (mb: number) => (mb / 1024).toFixed(1)
</script>

<header>
  <span class="brand">diffusers<span class="accent">-workflow</span></span>
  <nav>
    <a href="#/workflows" class:active={route.parts[0] === 'workflows'}>Workflows</a>
    <a href="#/jobs" class:active={route.parts[0] === 'jobs'}>Jobs</a>
    <a href="#/edit" class:active={route.parts[0] === 'edit'}>New</a>
    <a href="#/gallery" class:active={route.parts[0] === 'gallery'}>Gallery</a>
  </nav>
  <span class="vram muted">
    {#if memory?.info?.gpu_available}
      {memory.info.gpu_device_name} · {gb(memory.info.gpu_memory_allocated_mb ?? 0)} /
      {gb(memory.info.gpu_memory_total_mb ?? 0)} GB
    {:else}
      worker idle
    {/if}
  </span>
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
  nav { display: flex; gap: 1rem; flex: 1; }
  nav a { color: var(--muted); font-weight: 600; padding: 0.2rem 0; }
  nav a.active { color: var(--ink); border-bottom: 2px solid var(--accent); }
  .vram { font-size: 0.8rem; }
  main { max-width: 1100px; margin: 0 auto; padding: 1.2rem; }
</style>
