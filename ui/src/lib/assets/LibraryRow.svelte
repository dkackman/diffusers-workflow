<script lang="ts">
  import { Upload } from '@lucide/svelte'
  import GroupHeading from '../ui/GroupHeading.svelte'
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
  <GroupHeading
    size="library"
    label={section.label}
    count={section.assets.length}
    collapsed={!open}
    {ontoggle}
    title={open ? 'collapse this library' : 'expand this library'}
  />
  <span class="path muted">{section.root}</span>
  <span class="flex"></span>
  {#if !section.writable}
    <span class="muted" title="read-only: this server cannot write to it">
      read-only
    </span>
  {:else if section.origin === 'common' && shared}
    <!-- Only on the Shared assets page: a workspace's page shows just the
         shared assets its workflows use, so an upload there would vanish
         from the section it was made in -->
    <button
      class="withicon"
      onclick={() => onupload('shared')}
      disabled={busy}
      title="lands in the shared asset library - visible from every workspace under this root and cannot be moved afterwards"
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
  .path {
    font-family: var(--font-mono);
    word-break: break-all;
  }
</style>
