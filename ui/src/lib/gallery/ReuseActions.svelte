<script lang="ts">
  // The viewer's actions that take an output further, shared by the Gallery
  // and the job page: its star, Save as asset, Use as input… (a picture or
  // a clip) and the reference a workflow argument takes. Each page runs
  // the star and the save through its own action()
  import { Bookmark, ImagePlus, Star } from '@lucide/svelte'
  import type { MediaFile } from '../media'
  import { OUTPUT } from '../references'
  import ReferenceChip from '../ui/ReferenceChip.svelte'

  let {
    file,
    starred,
    saving,
    onstar,
    onsave,
    onuse,
  }: {
    file: MediaFile
    /** Null while the file's state is still being read: the star waits */
    starred: boolean | null
    /** A save is under way */
    saving: boolean
    onstar: () => void
    onsave: () => void
    onuse: () => void
  } = $props()
</script>

<button
  type="button"
  class="quiet icon"
  onclick={onstar}
  disabled={starred === null}
  aria-pressed={starred ?? undefined}
  aria-label={starred ? 'unstar this file' : 'star this file'}
  title={starred === null
    ? 'reading whether this file is starred'
    : starred
      ? 'starred - press S to unstar'
      : 'star this file - press S'}
>
  <Star size={16} fill={starred ? 'currentColor' : 'none'} />
</button>
<button
  type="button"
  class="quiet withicon"
  onclick={onsave}
  disabled={saving}
  title="save this file to the asset library, under a name later workflows can use"
>
  <Bookmark size={14} />Save as asset
</button>
{#if file.kind === 'image' || file.kind === 'video'}
  <button
    type="button"
    class="quiet withicon"
    onclick={onuse}
    title="open a workflow of this workspace with this file filled in"
  >
    <ImagePlus size={14} />Use as input…
  </button>
{/if}
<ReferenceChip prefix={OUTPUT} name={file.name} />
