<script lang="ts">
  // The job page's viewer: one of the job's outputs large, stepping through
  // them in the Results order, with what takes it further (its star, Save
  // as asset, Use as input…, its reference), a download, and "Note this
  // file" for a finished take of a request item; beside it what made it
  // (RecipePanel, its one read, which also says whether the file is
  // starred: the star waits for it). The open file is open= in the job route's
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
  import { isStarKey } from '../mediaViewer'
  import { leafName } from '../names'
  import { STAR_CHANGED } from '../picks.svelte'
  import { notify } from '../toast'
  import MediaViewer from '../ui/MediaViewer.svelte'
  import RecipePanel from '../ui/RecipePanel.svelte'
  import { workspace } from '../workspace.svelte'

  let {
    items,
    open = $bindable(null),
    onnote,
  }: {
    items: MediaFile[]
    open: string | null
    onnote?: (file: string) => void
  } = $props()

  // Whether each file is starred, as RecipePanel's read says, and the stars
  // set from here over it. Null until read: a toggle whose prior state is a
  // guess would offer an Undo that unstars a file starred elsewhere
  let starRead = $state<Record<string, boolean>>({})
  let starSet = $state<Record<string, boolean>>({})
  const starredNow = (name: string): boolean | null =>
    starSet[name] ?? starRead[name] ?? null
  function toggleStar(name: string) {
    const now = starredNow(name)
    if (now !== null) starring.run(name, !now)
  }
  // S stars or unstars the open file, as the star's title says
  function starKey(e: KeyboardEvent, file: MediaFile): boolean {
    if (!isStarKey(e)) return false
    toggleStar(file.name)
    return true
  }
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
      revert: async () => {
        if (starredNow(done.name) !== done.next)
          return notify.error(STAR_CHANGED)
        return writeStar(done.name, !done.next)
      },
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
    onstar={() => toggleStar(file.name)}
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
    <RecipePanel
      name={file.name}
      source="output"
      workspace={file.workspace}
      onread={(m) => {
        // A read is the server's word: it outranks what this page wrote
        starRead[file.name] = m.starred === true
        delete starSet[file.name]
      }}
    />
  {/key}
{/snippet}

<UseAsInput file={useInput} bind:open={useInputOpen} />

<MediaViewer {items} bind:open {actions} {aside} onkey={starKey} />
