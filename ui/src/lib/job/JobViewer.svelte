<script lang="ts">
  // The job page's viewer: one of the job's outputs large, stepping through
  // them in the Results order, with what takes it further (its star, Save
  // as asset, Use as input…, its reference), a download, and "Note this
  // file" for a finished take of a request item; beside it what made it
  // (RecipePanel, its one read). The open file is open= in the job route's
  // hash, written the way Back expects (mediaViewer.ts)
  import { tick } from 'svelte'
  import { FilePen } from '@lucide/svelte'
  import { action } from '../action.svelte'
  import { api } from '../api'
  import DownloadLink from '../DownloadLink.svelte'
  import ReuseActions from '../gallery/ReuseActions.svelte'
  import { saveAsAsset } from '../gallery/saveAsAsset'
  import UseAsInput from '../gallery/UseAsInput.svelte'
  import type { MediaFile } from '../media'
  import { leafName } from '../names'
  import MediaViewer from '../ui/MediaViewer.svelte'
  import RecipePanel from '../ui/RecipePanel.svelte'
  import { workspace } from '../workspace.svelte'

  let {
    items,
    open = $bindable(null),
    starredAtRead,
    onnote,
  }: {
    items: MediaFile[]
    open: string | null
    /** Whether a file was starred when its metadata was read, if it was */
    starredAtRead: (name: string) => boolean
    onnote?: (file: string) => void
  } = $props()

  // The stars set from here, over what the read said
  let starSet = $state<Record<string, boolean>>({})
  const starredNow = (name: string) => starSet[name] ?? starredAtRead(name)
  async function writeStar(name: string, next: boolean) {
    await api.setStarred(name, next)
    starSet[name] = next
    return { name, next }
  }
  // A star is written in the open workspace, so its Undo is held to it
  const jobWorkspace = workspace.current
  const starring = action(writeStar, {
    doing: 'star that file',
    undo: (done) => ({
      message: `${done.next ? 'Starred' : 'Unstarred'} ${leafName(done.name)}`,
      revert: () => writeStar(done.name, !done.next),
      target: {
        now: () => workspace.current,
        away: `Nothing undone: ${jobWorkspace} is not the open workspace now.`,
      },
    }),
  })
  const saving = action(saveAsAsset, {
    doing: 'save it as an asset',
    success: (r) => `Saved as asset ${r.name}`,
  })
  let useInput = $state<MediaFile | null>(null)
  let useInputOpen = $state(false)

  // The note box is on the page, under the sheet: close first
  async function noteThis(name: string) {
    open = null
    await tick()
    onnote?.(name)
  }
</script>

{#snippet actions(file: MediaFile)}
  <ReuseActions
    {file}
    starred={starredNow(file.name)}
    saving={saving.busy}
    onstar={() => starring.run(file.name, !starredNow(file.name))}
    onsave={() => saving.run(file)}
    onuse={() => {
      useInput = file
      useInputOpen = true
    }}
  />
  <DownloadLink href={api.outputDownloadUrl(file.name, file.workspace)} />
  {#if onnote}
    <button
      type="button"
      class="quiet withicon"
      aria-label="Note this file ({leafName(file.name)}) for the agent"
      onclick={() => noteThis(file.name)}
    >
      <FilePen size={14} />Note this file
    </button>
  {/if}
{/snippet}

{#snippet aside(file: MediaFile)}
  {#key file.name}
    <RecipePanel name={file.name} source="output" workspace={file.workspace} />
  {/key}
{/snippet}

<UseAsInput file={useInput} bind:open={useInputOpen} />

<MediaViewer {items} bind:open {actions} {aside} />
