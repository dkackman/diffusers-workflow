<script lang="ts">
  import { onDestroy } from 'svelte'
  import { Images, Upload } from '@lucide/svelte'
  import MediaPicker from '../assets/MediaPicker.svelte'
  import { reportUploads, uploadAll } from '../assets/uploads'
  import ErrorLine from '../ui/ErrorLine.svelte'
  import Suggest from '../ui/Suggest.svelte'
  import {
    assetLibrary,
    assetNamed,
    ensureAssetLibrary,
    loadAssetLibrary,
  } from '../assetlib.svelte'
  import { referenceCompletions } from '../refcomplete.svelte'
  import { ASSET, OUTPUT, referenceName } from '../references'
  import { ensureOutputLibrary, outputNamed } from '../outputlib.svelte'
  import type { MediaKind } from '../mediaArguments'
  import { dropFiles } from '../dropFiles'
  import { action } from '../action.svelte'

  let {
    id,
    location,
    kind,
    onchange,
  }: {
    id: string
    /** The argument's current file path or URL, already unwrapped from
     * whatever shape the JSON value carries it in. */
    location: string
    kind: MediaKind
    onchange: (location: string) => void
  } = $props()

  let fileInput: HTMLInputElement | undefined = $state()
  let browsing = $state(false)
  /** Our own word on an upload that did not land */
  let error = $state('')
  /** An upload that failed, as caught: shown through ErrorLine */
  let failure = $state<unknown>(null)
  // A locally-generated preview URL (object URL for a just-picked file, or
  // the location itself when it is already web-reachable). Cleaned up when
  // replaced so picking several files in a row doesn't leak object URLs.
  let localPreview = $state('')

  // An asset: or output: reference previews the file it names, once the
  // listing that says where that file is served has loaded
  const outputName = $derived(referenceName(location, OUTPUT))
  $effect(() => {
    if (outputName) ensureOutputLibrary()
  })
  // An asset: reference likewise, from the asset library
  // listing that says where that file is served has loaded
  const assetName = $derived(referenceName(location, ASSET))
  $effect(() => {
    if (assetName) ensureAssetLibrary()
  })
  const previewSrc = $derived(
    localPreview ||
      (assetName ? (assetNamed(assetName)?.url ?? '') : '') ||
      (outputName ? (outputNamed(outputName)?.url ?? '') : '') ||
      (location.startsWith('http://') ||
      location.startsWith('https://') ||
      location.startsWith('/')
        ? location
        : ''),
  )

  function revokeBlob(url: string) {
    if (url.startsWith('blob:')) URL.revokeObjectURL(url)
  }

  function setLocalPreview(url: string) {
    if (localPreview.startsWith('blob:')) URL.revokeObjectURL(localPreview)
    localPreview = url
  }

  // A blob URL pins the picked file's bytes until it is revoked; a failed
  // upload or navigating away mid-upload would otherwise leak it
  onDestroy(() => {
    if (localPreview.startsWith('blob:')) URL.revokeObjectURL(localPreview)
  })

  function pick(e: Event & { currentTarget: HTMLInputElement }) {
    const file = e.currentTarget.files?.[0]
    e.currentTarget.value = ''
    if (file) void uploading.run(file)
  }

  // A failed upload is said beside the field (ErrorLine), so the action
  // has nothing to announce: it is the busy guard
  const uploading = action(uploadFile, { doing: 'upload that file' })
  async function uploadFile(file: File) {
    error = ''
    failure = null
    const previous = localPreview
    localPreview = URL.createObjectURL(file)
    try {
      // This uploads to the workspace library, so only its names are taken
      const taken = new Set(
        assetLibrary.items
          ?.filter((a) => a.origin === 'workspace')
          .map((a) => a.name),
      )
      const { uploaded, skipped, failed } = await uploadAll([file], {
        shared: false,
        taken,
      })
      if (failed.length) throw failed[0].error
      const result = uploaded[0]
      if (!result) {
        // Skipped: the existing asset was left alone, so the field and its
        // preview stay as they were
        revokeBlob(localPreview)
        localPreview = previous
        if (skipped.length) reportUploads({ uploaded, skipped, failed })
        else error = 'unsupported file type'
        return
      }
      revokeBlob(previous)
      onchange(result.reference ?? result.url)
      localPreview = result.url
      // So the upload is among what an asset: completion offers next
      void loadAssetLibrary()
    } catch (err) {
      revokeBlob(previous)
      failure = err
    }
  }
</script>

<MediaPicker
  bind:open={browsing}
  {kind}
  current={location}
  onpick={(value) => {
    setLocalPreview('')
    onchange(value)
  }}
/>

<div
  class="media"
  use:dropFiles={{
    ondrop: (files) => void uploading.run(files[0]),
    accept: (f) => f.type.startsWith(`${kind}/`),
    disabled: uploading.busy,
  }}
>
  <div class="row1">
    <Suggest
      {id}
      class="mediafield"
      placeholder={`${ASSET}name, ${kind} path or URL`}
      value={location}
      complete={(text) => referenceCompletions(text, kind)}
      onchange={(value) => {
        setLocalPreview('')
        onchange(value)
      }}
    />
    <button
      type="button"
      class="quiet icon"
      title={`choose ${kind === 'image' ? 'an image' : 'a video'} from the assets or earlier jobs`}
      aria-label={`choose ${kind === 'image' ? 'an image' : 'a video'} from the assets or earlier jobs`}
      onclick={() => (browsing = true)}
    >
      <Images size={14} />
    </button>
    <button
      type="button"
      class="quiet icon"
      title={`browse for a local ${kind} file, or drop a file here`}
      aria-label={`browse for a local ${kind} file`}
      disabled={uploading.busy}
      onclick={() => fileInput?.click()}
    >
      <Upload size={14} />
    </button>
    <input
      bind:this={fileInput}
      class="hidden"
      type="file"
      accept={`${kind}/*`}
      onchange={pick}
    />
  </div>
  {#if uploading.busy}
    <div class="muted status">uploading…</div>
  {:else if failure != null}
    <div class="status">
      <ErrorLine error={failure} doing="upload that file" />
      <p class="muted">The path above can still be typed by hand.</p>
    </div>
  {:else if error}
    <div class="warn status">
      {error} - path above can still be typed by hand
    </div>
  {/if}
  {#if previewSrc}
    <div class="preview">
      {#if kind === 'image'}
        <img src={previewSrc} alt="" />
      {:else}
        <video src={previewSrc} muted controls></video>
      {/if}
    </div>
  {/if}
</div>

<style>
  .media {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    min-width: 0;
  }
  .row1 {
    display: flex;
    gap: 0.35rem;
    align-items: center;
  }
  .row1 :global(input.mediafield) {
    flex: 1;
    min-width: 0;
  }
  .hidden {
    display: none;
  }
  .icon {
    display: inline-flex;
    align-items: center;
    padding: 0.4rem 0.5rem;
    flex: none;
  }
  .status {
    font-size: 0.75rem;
  }
  .status p {
    margin: 0;
  }
  .preview img,
  .preview video {
    max-width: 160px;
    max-height: 120px;
    border-radius: 4px;
    border: 1px solid var(--line);
    display: block;
  }
</style>
