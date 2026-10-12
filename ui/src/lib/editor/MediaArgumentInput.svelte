<script lang="ts">
  import { onDestroy } from 'svelte'
  import { Images, Upload } from '@lucide/svelte'
  import AssetPicker from '../assets/AssetPicker.svelte'
  import { api } from '../api'
  import Suggest from '../ui/Suggest.svelte'
  import {
    assetNamed,
    ensureAssetLibrary,
    loadAssetLibrary,
  } from '../assetlib.svelte'
  import { referenceCompletions } from '../refcomplete.svelte'
  import { ASSET, reference, referenceName } from '../references'
  import type { MediaKind } from '../editor'

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
  let uploading = $state(false)
  let browsing = $state(false)
  let error = $state('')
  // A locally-generated preview URL (object URL for a just-picked file, or
  // the location itself when it is already web-reachable). Cleaned up when
  // replaced so picking several files in a row doesn't leak object URLs.
  let localPreview = $state('')

  // An asset: reference previews the file it names, once the library
  // listing that says where that file is served has loaded
  const assetName = $derived(referenceName(location, ASSET))
  $effect(() => {
    if (assetName) ensureAssetLibrary()
  })
  const previewSrc = $derived(
    localPreview ||
      (assetName ? (assetNamed(assetName)?.url ?? '') : '') ||
      (location.startsWith('http://') ||
      location.startsWith('https://') ||
      location.startsWith('/')
        ? location
        : ''),
  )

  function setLocalPreview(url: string) {
    if (localPreview.startsWith('blob:')) URL.revokeObjectURL(localPreview)
    localPreview = url
  }

  // A blob URL pins the picked file's bytes until it is revoked; a failed
  // upload or navigating away mid-upload would otherwise leak it
  onDestroy(() => {
    if (localPreview.startsWith('blob:')) URL.revokeObjectURL(localPreview)
  })

  async function pick(e: Event & { currentTarget: HTMLInputElement }) {
    const file = e.currentTarget.files?.[0]
    e.currentTarget.value = ''
    if (!file) return

    error = ''
    setLocalPreview(URL.createObjectURL(file))
    uploading = true
    try {
      const result = await api.uploadMedia(file)
      onchange(result.reference ?? result.url)
      setLocalPreview(result.url)
      // So the upload is among what an asset: completion offers next
      void loadAssetLibrary()
    } catch (err) {
      error = err instanceof Error ? err.message : 'upload failed'
    } finally {
      uploading = false
    }
  }
</script>

<AssetPicker
  bind:open={browsing}
  {kind}
  current={assetName}
  onpick={(name) => {
    setLocalPreview('')
    onchange(reference(ASSET, name))
  }}
/>

<div class="media">
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
      title={`choose ${kind === 'image' ? 'an image' : 'a video'} from the asset library`}
      aria-label={`choose ${kind === 'image' ? 'an image' : 'a video'} from the asset library`}
      onclick={() => (browsing = true)}
    >
      <Images size={14} />
    </button>
    <button
      type="button"
      class="quiet icon"
      title={`browse for a local ${kind} file`}
      aria-label={`browse for a local ${kind} file`}
      disabled={uploading}
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
  {#if uploading}
    <div class="muted status">uploading…</div>
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
  .warn {
    color: var(--warn);
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
