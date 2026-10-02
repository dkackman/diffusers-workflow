<script lang="ts">
  import { ChevronDown, ChevronRight, Upload } from '@lucide/svelte'
  import type { AssetLibrary } from '../types'

  // A library section's header: the label, the count, the root it reads,
  // and the one action that library offers
  let {
    section,
    open,
    shared,
    busy,
    ontoggle,
    onupload,
  }: {
    section: AssetLibrary & { label: string; assets: unknown[] }
    open: boolean
    shared: boolean
    busy: boolean
    ontoggle: () => void
    onupload: (target: 'workspace' | 'shared') => void
  } = $props()
</script>

<div class="libraryrow">
  <button
    class="library"
    onclick={ontoggle}
    title={open ? 'collapse this library' : 'expand this library'}
  >
    {#if open}<ChevronDown size={15} />{:else}<ChevronRight size={15} />{/if}
    {section.label}
    <span class="muted">({section.assets.length})</span>
  </button>
  <span class="path muted">{section.root}</span>
  <span class="flex"></span>
  {#if !section.writable}
    <span class="muted" title="read-only: this server cannot write to it"
      >read-only</span
    >
  {:else if section.origin === 'common'}
    <button
      class="withicon"
      class:quiet={!shared}
      onclick={() => onupload('shared')}
      disabled={busy}
      title="lands in the shared library - visible from every workspace under this root and cannot be moved afterwards"
    >
      <Upload size={14} />Upload to shared
    </button>
  {:else if section.origin === 'workspace'}
    <button
      class="withicon"
      onclick={() => onupload('workspace')}
      disabled={busy}
    >
      <Upload size={14} />Upload
    </button>
  {/if}
</div>

<style>
  /* A library section header: the label, the count, the root it reads,
     and the one action that library offers */
  .libraryrow {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
    margin: 1.6rem 0 var(--space-2);
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--line);
  }
  /* A library is a heavier folder heading: the same mono name the engine
     resolves, ink rather than muted, because it is the page's top level */
  .library {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    background: none;
    border: none;
    color: var(--ink);
    font-family: var(--font-mono);
    font-weight: 600;
    font-size: var(--t-md);
    padding: 0;
    margin: 0;
    cursor: pointer;
  }
  .library:hover {
    filter: none;
    color: var(--accent);
  }
  .path {
    font-family: var(--font-mono);
    word-break: break-all;
  }
</style>
