<script lang="ts">
  // The one input for a file or a reference: typed (with the earlier steps'
  // results offered first, then the asset and output libraries), chosen by
  // eye (MediaPicker), uploaded or dropped. The file it names is shown as a
  // tile; a value it cannot show keeps its text and shows the kind's glyph
  // or nothing, and is never rewritten. An upload belongs to the value it
  // started from: one that lands after the field moved on is reported, not
  // written. Choices of what to offer and show are referenceField.ts's.
  import { onDestroy, type Snippet } from 'svelte'
  import { Images, Upload } from '@lucide/svelte'
  import MediaPicker from '../assets/MediaPicker.svelte'
  import {
    reportUploads,
    uploadAll,
    type UploadResult,
  } from '../assets/uploads'
  import ErrorLine from './ErrorLine.svelte'
  import OutputTile from './OutputTile.svelte'
  import Suggest from './Suggest.svelte'
  import {
    assetLibrary,
    ensureAssetLibrary,
    loadAssetLibrary,
  } from '../assetlib.svelte'
  import { ensureOutputLibrary } from '../outputlib.svelte'
  import { ASSET, OUTPUT, isReference, referenceName } from '../references'
  import {
    fieldRows,
    kindWithArticle,
    mediaKindOf,
    missingOf,
    previewOf,
    type FieldKind,
  } from '../referenceField'
  import type { MediaFile } from '../media'
  import { dropFiles } from '../dropFiles'
  import { action } from '../action.svelte'
  import { notify } from '../toast'

  let {
    id,
    value,
    kind = undefined,
    earlier = undefined,
    placeholder = undefined,
    onchange,
    oninput = undefined,
    children = undefined,
  }: {
    id: string
    /** The file path, URL or reference the field holds */
    value: string
    kind?: FieldKind
    /** The earlier steps' results this field can take, offered first */
    earlier?: string[]
    placeholder?: string
    onchange: (value: string) => void
    oninput?: (value: string) => void
    /** More buttons, after Choose… and Upload… */
    children?: Snippet
  } = $props()

  const media = $derived(mediaKindOf(kind))
  const named = $derived(kindWithArticle(kind))
  /** Choose… and Upload… belong to a field of a kind, or one holding a
   * file's reference; not to a kindless field holding a variable, a prompt
   * or a step's result */
  const takesFiles = $derived(
    kind !== undefined || /^(asset:|output:|https?:\/\/|\/)/.test(value),
  )

  let fileInput: HTMLInputElement | undefined = $state()
  let browsing = $state(false)
  /** Our own word on a file that was not taken */
  let error = $state('')
  /** An upload that failed, as caught: shown through ErrorLine */
  let failure = $state<unknown>(null)
  /** A picked file's own picture, for the value it belongs to: shown while
   * it uploads, and after, until the field holds something else */
  let local = $state<{ for: string; file: MediaFile } | null>(null)
  let destroyed = false
  /** What the person has typed and not yet committed, for the value it was
   * typed over: Suggest keeps it until change or blur */
  let typed = $state<{ over: string; text: string } | null>(null)
  const live = () => (typed?.over === value ? typed.text : value)

  function typing(text: string) {
    typed = { over: value, text }
    oninput?.(text)
  }

  // A library reference previews once the listing that holds it has loaded
  $effect(() => {
    if (referenceName(value, ASSET) !== null) ensureAssetLibrary()
    if (referenceName(value, OUTPUT) !== null) ensureOutputLibrary()
  })
  const preview = $derived(
    (local?.for === value ? local.file : null) ??
      previewOf(value, kind) ??
      missingOf(value, kind),
  )

  function setLocal(next: typeof local) {
    const old = local?.file.url ?? ''
    if (old.startsWith('blob:') && old !== next?.file.url)
      URL.revokeObjectURL(old)
    local = next
  }

  // A blob URL pins the picked file's bytes until it is revoked; a failed
  // upload or navigating away mid-upload would otherwise leak it
  onDestroy(() => {
    destroyed = true
    setLocal(null)
  })

  function commit(next: string) {
    setLocal(null)
    error = ''
    onchange(next)
  }

  /** The file as the field shows it while it uploads */
  function pickedMedia(file: File, url: string): MediaFile {
    const kindOf = file.type.split('/', 1)[0]
    const shown = kindOf === 'video' || kindOf === 'audio' ? kindOf : 'image'
    return {
      name: file.name,
      kind: shown,
      url,
      label: file.name,
      workspace: '',
      thumb: kindOf === 'image' ? url : null,
    }
  }

  function take(file: File | undefined) {
    if (!file) return
    if (media && !file.type.startsWith(`${media}/`)) {
      failure = null
      error = `That isn't ${named} file.`
      return
    }
    void uploading.run(file)
  }

  function pick(e: Event & { currentTarget: HTMLInputElement }) {
    const file = e.currentTarget.files?.[0]
    e.currentTarget.value = ''
    take(file)
  }

  // A failed upload is said beside the field (ErrorLine), so the action
  // has nothing to announce: it is the busy guard
  const uploading = action(uploadFile, { doing: 'upload that file' })
  async function uploadFile(file: File) {
    error = ''
    failure = null
    const from = live()
    const previous = local
    local = { for: value, file: pickedMedia(file, URL.createObjectURL(file)) }
    try {
      // This uploads to the workspace library, so only its names are taken
      const taken = new Set(
        assetLibrary.items
          ?.filter((a) => a.origin === 'workspace')
          .map((a) => a.name),
      )
      const result = await uploadAll([file], { shared: false, taken })
      landed(result, from, previous)
    } catch (err) {
      // The file never uploaded, so its picture goes with it
      setLocal(previous)
      failure = err
    }
    // A picture the field no longer shows lets go of its bytes
    const old = previous?.file.url ?? ''
    if (old.startsWith('blob:') && local?.file.url !== old)
      URL.revokeObjectURL(old)
  }

  /** Where a finished upload goes: into the field when it still holds the
   * value the upload started from, else only into the library */
  function landed(result: UploadResult, from: string, previous: typeof local) {
    const { uploaded, skipped, failed } = result
    if (failed.length) throw failed[0].error
    const done = uploaded[0]
    if (!done) {
      // Skipped: the existing asset was left alone, so the field and its
      // preview stay as they were
      setLocal(previous)
      if (skipped.length) reportUploads(result)
      else
        error =
          'unsupported file type - the path above can still be typed by hand'
      return
    }
    // So the upload is among what an asset: completion offers next
    void loadAssetLibrary()
    const next = done.reference ?? done.url
    if (destroyed || live() !== from) {
      setLocal(null)
      notify.success(
        `Uploaded as ${next}. The field changed meanwhile, so it was left as it is.`,
      )
      return
    }
    if (local) local = { for: next, file: local.file }
    onchange(next)
  }
</script>

<MediaPicker
  bind:open={browsing}
  kind={media}
  current={value}
  onpick={commit}
/>

<div
  class="reference"
  use:dropFiles={{
    ondrop: (files) => take(files[0]),
    disabled: uploading.busy,
  }}
>
  <div class="row1">
    <Suggest
      {id}
      class={isReference(value) ? 'referencefield ref' : 'referencefield'}
      placeholder={placeholder ?? `type, drop or choose ${named}`}
      {value}
      complete={(text) => fieldRows(text, kind, earlier ?? [])}
      onchange={commit}
      oninput={typing}
    />
    {#if takesFiles}
      <button
        type="button"
        class="quiet withicon"
        title={`Choose ${named} from the assets or earlier jobs`}
        onclick={() => (browsing = true)}
      >
        <Images size={14} /> Choose…
      </button>
      <button
        type="button"
        class="quiet withicon"
        title={`Upload ${named} from this computer, or drop one here`}
        disabled={uploading.busy}
        onclick={() => fileInput?.click()}
      >
        <Upload size={14} /> Upload…
      </button>
      <input
        bind:this={fileInput}
        class="hidden"
        type="file"
        accept={media ? `${media}/*` : undefined}
        onchange={pick}
      />
    {/if}
    {@render children?.()}
  </div>
  {#if uploading.busy}
    <div class="muted status">uploading…</div>
  {:else if failure != null}
    <div class="status">
      <ErrorLine error={failure} doing="upload that file" />
      <p class="muted">The path above can still be typed by hand.</p>
    </div>
  {:else if error}
    <div class="warn status">{error}</div>
  {/if}
  {#if preview}
    <div class="preview">
      <OutputTile file={preview} size="strip" />
    </div>
  {/if}
</div>

<style>
  .reference {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    min-width: 0;
  }
  .row1 {
    display: flex;
    gap: 0.35rem;
    align-items: center;
    flex-wrap: wrap;
  }
  .row1 :global(input.referencefield) {
    flex: 1;
    min-width: 10rem;
  }
  .hidden {
    display: none;
  }
  .withicon {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    flex: none;
  }
  .status {
    font-size: 0.75rem;
  }
  .status p {
    margin: 0;
  }
</style>
