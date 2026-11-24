<script lang="ts">
  // A file as a tile: the one place a listed file is drawn small (the
  // Gallery, the asset library, the shadowed entries). One look per kind - a
  // picture is its thumbnail, letterboxed and never cropped; a clip is its
  // poster with the play badge and its length, never a <video>, so a tile
  // never opens a stream; audio is its glyph, name and length; text is its
  // glyph and first lines, read once when the tile is first seen. The look
  // is `.filetile` in patterns.css; the glyphs are media.ts's.
  import type { Snippet } from 'svelte'
  import { Film, Image, Pencil, Play, Star } from '@lucide/svelte'
  import { api } from '../api'
  import { formatDuration } from '../format'
  import { hasPicture, kindGlyph, type MediaFile } from '../media'
  import { resource } from '../resource.svelte'

  let {
    file,
    size = 'grid',
    caption = 'line',
    href,
    onclick,
    active = false,
    picked = false,
    dim = false,
    alt,
    badges,
    meta,
    title,
    'aria-label': ariaLabel,
    'aria-current': ariaCurrent,
  }: {
    file: MediaFile
    size?: 'grid' | 'card' | 'strip' | 'take' | 'mini' | 'row'
    caption?: 'none' | 'line' | 'clamp'
    href?: string
    onclick?: () => void
    active?: boolean
    picked?: boolean
    dim?: boolean
    alt?: string
    badges?: Snippet
    meta?: Snippet
    /** The tile's own title and accessible name, when its content is not */
    title?: string
    'aria-label'?: string
    /** The one a picker's field holds now */
    'aria-current'?: 'true'
  } = $props()

  /** How many of a text file's lines the tile shows */
  const EXCERPT_LINES = 3

  // A thumbnail the server could not make: the kind's glyph stands in, until
  // the tile is given another
  let broken = $state<string | null>(null)
  const picture = $derived(
    !!file.thumb && broken !== file.thumb && hasPicture(file.kind),
  )
  const duration = $derived(
    file.duration != null && file.kind !== 'image'
      ? formatDuration(file.duration)
      : '',
  )
  const noteLine = $derived(file.note ? file.note.split('\n')[0] : '')

  // A text file's first lines, read when the tile first scrolls into view
  // (a grid of hundreds never reads the ones nobody scrolls to), and again
  // only when the file's stamp moves - a Replace, a rewrite - never because
  // a listing was read again. Deriveds, so a new object for the same file
  // is no new question
  let seen = $state(false)
  const url = $derived(file.url)
  const stamp = $derived(file.stamp ?? '')
  const home = $derived(file.workspace)
  let face = $state<HTMLElement | null>(null)
  const excerpt = textExcerpt()

  function textExcerpt() {
    // A row is a glyph; it reads nothing
    if (file.kind !== 'text' || size === 'row') return null
    return resource(
      async () => {
        // The stamp is part of the question, though the read needs only the URL
        const question = { url, stamp }
        if (!seen) return null
        const text = await api.fileText(question.url)
        if (!text.trim()) return '(empty)'
        return text.split('\n').slice(0, EXCERPT_LINES).join('\n')
      },
      { watch: [], workspace: () => home },
    )
  }

  $effect(() => {
    if (!excerpt || seen || !face) return
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

  const classes = $derived(['filetile', `tile-${size}`, `caption-${caption}`])
</script>

{#snippet glyph()}
  <span class="tile-glyph" aria-hidden="true">
    {#if kindGlyph(file.kind)}
      {kindGlyph(file.kind)}
    {:else if file.kind === 'video'}
      <Film size={20} strokeWidth={1.5} />
    {:else}
      <Image size={20} strokeWidth={1.5} />
    {/if}
  </span>
{/snippet}

{#snippet body()}
  <span class="tile-media" bind:this={face}>
    {#if picture}
      <img
        class="tile-pic"
        src={file.thumb}
        alt={alt ?? file.name}
        loading="lazy"
        onerror={() => (broken = file.thumb ?? null)}
      />
    {:else if file.kind === 'audio'}
      <span class="tile-sound">
        {@render glyph()}
        <!-- The caption carries the name; this copy is for the eye -->
        <span
          class="tile-name"
          aria-hidden={caption === 'none' ? undefined : 'true'}
        >
          {file.label}
        </span>
        {#if duration}<span class="tile-length">{duration}</span>{/if}
      </span>
    {:else if file.kind === 'text'}
      <span class="tile-text">
        {@render glyph()}
        {#if typeof excerpt?.data === 'string'}
          <!-- The file's lines are not its name: hidden from the name -->
          <span class="tile-excerpt" aria-hidden="true">{excerpt.data}</span>
        {/if}
      </span>
    {:else}
      {@render glyph()}
    {/if}
    {#if file.kind === 'video'}
      <span class="videobadge" aria-hidden="true"><Play size={11} /></span>
      {#if duration}<span class="tile-time">{duration}</span>{/if}
    {/if}
  </span>
  {#if caption !== 'none'}
    <span class="tile-caption" title={file.name}>{file.label}</span>
  {/if}
  {#if size !== 'row' && (badges || file.note || file.starred)}
    <span class="tile-marks">
      {@render badges?.()}
      {#if file.note}
        <span
          class="tile-mark note-mark"
          title={noteLine}
          aria-label="Has a note"
        >
          <Pencil size={11} />
        </span>
      {/if}
      {#if file.starred}
        <span class="tile-mark star-mark" aria-label="Starred">
          <Star size={11} fill="currentColor" />
        </span>
      {/if}
    </span>
  {/if}
  {#if meta && size !== 'row'}<span class="tile-meta">
      {@render meta()}
    </span>{/if}
{/snippet}

{#if href}
  <a
    class={classes}
    class:active
    class:picked
    class:dim
    {href}
    {onclick}
    {title}
    aria-label={ariaLabel}
    aria-current={ariaCurrent}
  >
    {@render body()}
  </a>
{:else if onclick}
  <button
    type="button"
    class={classes}
    class:active
    class:picked
    class:dim
    {onclick}
    {title}
    aria-label={ariaLabel}
    aria-current={ariaCurrent}
  >
    {@render body()}
  </button>
{:else}
  <div class={classes} class:picked class:dim {title} aria-label={ariaLabel}>
    {@render body()}
  </div>
{/if}
