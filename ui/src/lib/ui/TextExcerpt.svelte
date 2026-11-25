<script lang="ts">
  // A text file's first lines on its tile, read when the tile first scrolls
  // into view (a grid of hundreds never reads the ones nobody scrolls to),
  // and again only when the file's stamp moves - a Replace, a rewrite -
  // never because a listing was read again. OutputTile mounts it while its
  // file is text, so a tile handed another kind of file starts or stops
  // reading with it
  import { api } from '../api'
  import { resource } from '../resource.svelte'
  import type { MediaFile } from '../media'

  /** How many of a text file's lines the tile shows */
  const EXCERPT_LINES = 3

  let { file, face }: { file: MediaFile; face: HTMLElement | null } = $props()

  // Deriveds, so a new object for the same file is no new question
  let seen = $state(false)
  const url = $derived(file.url)
  const stamp = $derived(file.stamp ?? '')
  const home = $derived(file.workspace)

  const excerpt = resource(
    async () => {
      // The stamp is part of the question, though the read needs only the URL
      const question = { url, stamp }
      if (!seen) return null
      const text = await api.fileText(question.url)
      if (!text.trim()) return '(empty)'
      return text.split('\n').slice(0, EXCERPT_LINES).join('\n')
    },
    // Another file's lines never stand in for this one's
    { watch: [], workspace: () => home, subject: () => url },
  )

  $effect(() => {
    if (seen || !face) return
    if (typeof IntersectionObserver === 'undefined') {
      seen = true
      return
    }
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        seen = true
        observer.disconnect()
      }
    })
    observer.observe(face)
    return () => observer.disconnect()
  })
</script>

{#if typeof excerpt.data === 'string'}
  <!-- The file's lines are not its name: hidden from the name -->
  <span class="tile-excerpt" aria-hidden="true">{excerpt.data}</span>
{/if}
