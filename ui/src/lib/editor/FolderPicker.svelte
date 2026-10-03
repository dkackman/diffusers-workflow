<script lang="ts">
  import { NEW_FOLDER } from '../editorShell.svelte'

  // Which folder a save lands in: the root, an existing folder from the
  // listing, or a new one named in the field that appears beside it
  let {
    folder = $bindable(),
    newFolder = $bindable(),
    folders,
    id = undefined,
    newFolderTitle,
  }: {
    folder: string
    newFolder: string
    folders: string[]
    id?: string
    newFolderTitle: string
  } = $props()
</script>

<select {id} class="folderpick" bind:value={folder} title="folder to save into">
  <option value="">(root)</option>
  {#each folders as existing (existing)}<option value={existing}
      >{existing}/</option
    >{/each}
  <option value={NEW_FOLDER}>new folder…</option>
</select>
{#if folder === NEW_FOLDER}
  <input
    class="newfolder"
    bind:value={newFolder}
    placeholder="folder name"
    title={newFolderTitle}
  />
{/if}

<style>
  .folderpick {
    max-width: 160px;
  }
  .newfolder {
    max-width: 140px;
  }
</style>
