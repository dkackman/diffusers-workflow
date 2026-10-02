<script lang="ts">
  import type { Component } from 'svelte'
  import type { EditorView } from '../editorShell.svelte'

  // The editor's view picker: one segmented button per view, the current
  // one marked
  let {
    view,
    options,
    onselect,
  }: {
    view: EditorView
    options: {
      view: EditorView
      label: string
      title: string
      icon: Component<{ size?: number }>
    }[]
    onselect: (view: EditorView) => void
  } = $props()
</script>

<div class="viewswitch" role="group" aria-label="editor view">
  {#each options as option (option.view)}
    <button
      class="quiet withicon"
      class:activebtn={view === option.view}
      onclick={() => onselect(option.view)}
      title={option.title}
    >
      <option.icon size={14} />{option.label}
    </button>
  {/each}
</div>

<style>
  .viewswitch {
    display: inline-flex;
  }
  .viewswitch button {
    border-radius: 0;
  }
  .viewswitch button:first-child {
    border-radius: 6px 0 0 6px;
  }
  .viewswitch button:last-child {
    border-radius: 0 6px 6px 0;
  }
  .viewswitch button + button {
    margin-left: -1px;
  }
  .activebtn {
    border-color: var(--accent);
    color: var(--accent);
    position: relative;
    z-index: 1;
  }
</style>
