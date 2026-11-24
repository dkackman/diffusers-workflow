<script lang="ts">
  // One file drawn large: the only place besides OutputTile a media element
  // is written (`raw_media`). The viewer draws its file with it, and
  // Compare's synced transport takes each video through `onvideo`. An image
  // is the picture; a clip a <video> that loops, with its native controls
  // (tabindex -1, so a Tab never lands on it and the viewer keeps the arrow
  // keys); audio an <audio> with its controls; text the file's contents.
  // `fit` bounds the element by its box, never cropping it (`.media-fit`
  // in patterns.css).
  import { api } from '../api'
  import type { MediaFile } from '../media'
  import { resource } from '../resource.svelte'
  import ErrorLine from './ErrorLine.svelte'

  let {
    file,
    fit = true,
    onvideo,
    controls = true,
  }: {
    file: MediaFile
    fit?: boolean
    /** The video element as it mounts, and null as it goes */
    onvideo?: (el: HTMLVideoElement | null) => void
    /** The native controls; Compare drives its videos from one transport */
    controls?: boolean
  } = $props()

  let video = $state<HTMLVideoElement | null>(null)
  $effect(() => {
    if (!video || !onvideo) return
    const report = onvideo
    report(video)
    return () => report(null)
  })

  const url = $derived(file.url)
  const home = $derived(file.workspace)
  const isText = $derived(file.kind === 'text')
  const text = resource(
    async () => {
      const question = { url, isText }
      if (!question.isText) return null
      return api.fileText(question.url)
    },
    { watch: [], workspace: () => home },
  )

  const cls = $derived(['media-body', fit && 'media-fit'])
</script>

{#if file.kind === 'image'}
  <img class={cls} src={file.url} alt={file.label} draggable="false" />
{:else if file.kind === 'video'}
  <video
    bind:this={video}
    class={cls}
    src={file.url}
    {controls}
    loop
    playsinline
    tabindex="-1"
  >
    <!-- A generated clip comes with no captions: the track is empty -->
    <track kind="captions" />
  </video>
{:else if file.kind === 'audio'}
  <audio class="media-body media-sound" src={file.url} {controls}></audio>
{:else if text.error}
  <ErrorLine error={text.error} doing="read the file" />
{:else}
  <pre class="media-body media-text">{text.data ?? ''}</pre>
{/if}
