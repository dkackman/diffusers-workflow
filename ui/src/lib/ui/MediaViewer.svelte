<script lang="ts">
  // One file, large: the whole window, the file fitted to it (nothing
  // scrolls at fit), zoom to 8x with a drag to pan, previous and next
  // through the page's own list, a swipe at fit, full screen and keys. Every
  // key has a button. A page keeps the open file in the hash as open= and
  // binds `open`; what opening, stepping, closing and Back do to the
  // history is mediaViewer.ts's (its head says why). An `open` no item
  // names renders nothing - the page says the file is gone - but a file
  // that leaves `items` while it is shown stays, and steps on from where it
  // stood. The look is `.ui-sheet-fill` and `.viewer-*` in patterns.css.
  import { untrack, type Snippet } from 'svelte'
  import {
    ChevronLeft,
    ChevronRight,
    Expand,
    Maximize,
    Minimize,
    Scan,
    X,
    ZoomIn,
    ZoomOut,
  } from '@lucide/svelte'
  import { isEditable } from '../editable'
  import type { MediaFile } from '../media'
  import {
    SEEK_SECONDS,
    clampPan,
    nextZoom,
    openInHash,
    swipeStep,
    viewerKey,
    writeOpen,
    type ViewerKey,
  } from '../mediaViewer'
  import { leafName } from '../names'
  import { neighbor } from '../neighbor'
  import { overlayOpen } from './layers.svelte'
  import MediaBody from './MediaBody.svelte'
  import Modal from './Modal.svelte'

  let {
    items,
    open = $bindable(null),
    actions,
    aside,
    onkey,
  }: {
    items: MediaFile[]
    open: string | null
    /** The page's own buttons for the file, between full screen and Close */
    actions?: Snippet<[MediaFile]>
    /** What the page says about the file, beside the stage */
    aside?: Snippet<[MediaFile]>
    /** A key the viewer does not take; true when the page handled it */
    onkey?: (e: KeyboardEvent, file: MediaFile) => boolean
  } = $props()

  // The file last shown under `open`, and where it last stood in `items`:
  // a file that leaves the list while it is shown stays, and steps on from
  // there (neighbor.ts). Plain variables: memory, not state
  let held: MediaFile | null = null
  let lastAt = -1
  const names = $derived(items.map((item) => item.name))
  const file = $derived.by(() => {
    const at = open === null ? -1 : names.indexOf(open)
    if (at >= 0) {
      held = items[at]
      lastAt = at
      return held
    }
    return held && held.name === open ? held : null
  })
  const listedAt = $derived(file ? names.indexOf(file.name) : -1)
  const prevName = $derived(
    file ? neighbor(names, file.name, -1, lastAt) : null,
  )
  const nextName = $derived(file ? neighbor(names, file.name, 1, lastAt) : null)
  const leaf = $derived(file ? leafName(file.name) : '')

  // Every change of `open` goes into the hash; an `open` naming no file
  // writes nothing. Back (a hashchange) reads it back
  $effect(() => {
    const name = open
    untrack(() => {
      if (name === null || file) writeOpen(name)
    })
  })
  $effect(() => {
    const follow = () => {
      const named = openInHash()
      if (named !== open) open = named
    }
    window.addEventListener('hashchange', follow)
    return () => window.removeEventListener('hashchange', follow)
  })

  function close() {
    open = null
  }
  // Previous and Next at an end are aria-disabled, not disabled: a
  // disabled button drops its focus to <body>, outside the sheet's keys
  function step(to: string | null) {
    if (to) open = to
  }

  // Zoom is relative to fit (1 is fitted); the pan offsets the file's
  // centre from the stage's, in px
  let stage = $state<HTMLElement | null>(null)
  let zoom = $state(1)
  let pan = $state({ x: 0, y: 0 })
  const zoomable = $derived(file?.kind === 'image' || file?.kind === 'video')
  const fitted = $derived(zoom <= 1 + 1e-6)
  $effect(() => {
    void file?.name
    zoom = 1
    pan = { x: 0, y: 0 }
  })

  function mediaEl(): HTMLImageElement | HTMLVideoElement | null {
    return (
      stage?.querySelector<HTMLImageElement | HTMLVideoElement>(
        'img.media-body, video.media-body',
      ) ?? null
    )
  }
  /** The file's fitted size over its own: the zoom of actual size is 1/fit */
  function fitScale(): number {
    const el = mediaEl()
    if (!el) return 1
    const own = el instanceof HTMLImageElement ? el.naturalWidth : el.videoWidth
    return own > 0 && el.offsetWidth > 0 ? el.offsetWidth / own : 1
  }
  function setZoom(next: number) {
    zoom = Math.max(1, next)
    pan = heldPan(pan)
  }
  function heldPan(want: { x: number; y: number }) {
    const el = mediaEl()
    if (!stage || !el) return { x: 0, y: 0 }
    const box = { w: stage.clientWidth, h: stage.clientHeight }
    return clampPan(want, zoom, box, { w: el.offsetWidth, h: el.offsetHeight })
  }
  function zoomBy(by: 1 | -1) {
    const fit = fitScale()
    setZoom(nextZoom(zoom * fit, fit, by) / fit)
  }
  const fitToWindow = () => setZoom(1)
  const actualSize = () => setZoom(1 / fitScale())

  function seek(by: number) {
    const el = stage?.querySelector<HTMLMediaElement>('video, audio')
    if (el) el.currentTime = Math.max(0, el.currentTime + by)
  }

  // Full screen: the button is there only where the browser allows it
  const canFullscreen =
    typeof document !== 'undefined' && document.fullscreenEnabled === true
  let sheetFull = $state(false)
  $effect(() => {
    const sync = () => (sheetFull = document.fullscreenElement !== null)
    document.addEventListener('fullscreenchange', sync)
    return () => document.removeEventListener('fullscreenchange', sync)
  })
  function toggleFullscreen() {
    if (!canFullscreen) return
    if (document.fullscreenElement) void document.exitFullscreen()
    else void stage?.closest<HTMLElement>('.ui-sheet')?.requestFullscreen()
  }

  const ACTS: Record<ViewerKey, () => void> = {
    prev: () => step(prevName),
    next: () => step(nextName),
    zoomIn: () => zoomable && zoomBy(1),
    zoomOut: () => zoomable && zoomBy(-1),
    fit: fitToWindow,
    actual: () => zoomable && actualSize(),
    fullscreen: toggleFullscreen,
    seekBack: () => seek(-SEEK_SECONDS),
    seekOn: () => seek(SEEK_SECONDS),
  }

  /** A key typed in a field, or under a dialog opened above the viewer, is
   * not the viewer's. Its own video is no field: an arrow steps even after
   * a click on the clip (GA-15) */
  function notOurs(e: KeyboardEvent): boolean {
    const field =
      isEditable(e.target) && !(e.target instanceof HTMLMediaElement)
    return field || overlayOpen(1)
  }
  function onkeydowncapture(e: KeyboardEvent) {
    if (!file || notOurs(e)) return
    const act = viewerKey(e)
    if (act) ACTS[act]()
    const handled = act !== null || (onkey?.(e, file) ?? false)
    // handled, a focused video never also seeks on the arrow
    if (handled) e.preventDefault()
  }

  // A drag pans when zoomed; a quick flat stroke at fit steps. Only a
  // press on the stage, its frame or a picture: a drag on a clip's seek bar
  // scrubs and a drag over text selects, so a video, audio, text or control
  // never starts one
  let press: { x: number; y: number; t: number; pan: typeof pan } | null = null
  function gestureTarget(target: EventTarget | null): boolean {
    if (!(target instanceof Element)) return false
    if (target === stage || target instanceof HTMLImageElement) return true
    return target.classList.contains('viewer-frame')
  }
  function onpointerdown(e: PointerEvent) {
    press = gestureTarget(e.target)
      ? { x: e.clientX, y: e.clientY, t: e.timeStamp, pan }
      : null
  }
  function onpointermove(e: PointerEvent) {
    if (!press || fitted) return
    const want = {
      x: press.pan.x + e.clientX - press.x,
      y: press.pan.y + e.clientY - press.y,
    }
    pan = heldPan(want)
  }
  function onpointerup(e: PointerEvent) {
    const was = press
    press = null
    if (!was || !fitted) return
    const by = swipeStep(
      e.clientX - was.x,
      e.clientY - was.y,
      e.timeStamp - was.t,
    )
    if (by) step(by === 1 ? nextName : prevName)
  }
  function ondblclick() {
    if (!zoomable) return
    if (fitted) actualSize()
    else fitToWindow()
  }
</script>

{#if file}
  <Modal
    bind:open={() => true, (shown) => !shown && close()}
    label={leaf}
    fill
    {onkeydowncapture}
  >
    <div class="viewer-bar">
      <span class="viewer-name mono" title={file.name}>{leaf}</span>
      {#if listedAt >= 0}
        <span class="num muted">{listedAt + 1} of {items.length}</span>
      {/if}
      <button
        type="button"
        class="quiet icon"
        aria-label="Previous file"
        title="Previous file"
        aria-disabled={!prevName}
        onclick={() => step(prevName)}
      >
        <ChevronLeft size={16} />
      </button>
      <button
        type="button"
        class="quiet icon"
        aria-label="Next file"
        title="Next file"
        aria-disabled={!nextName}
        onclick={() => step(nextName)}
      >
        <ChevronRight size={16} />
      </button>
      <button
        type="button"
        class="quiet icon"
        aria-label="Zoom out"
        title="Zoom out"
        disabled={!zoomable || fitted}
        onclick={() => zoomBy(-1)}
      >
        <ZoomOut size={16} />
      </button>
      <button
        type="button"
        class="quiet icon"
        aria-label="Zoom in"
        title="Zoom in"
        disabled={!zoomable}
        onclick={() => zoomBy(1)}
      >
        <ZoomIn size={16} />
      </button>
      <button
        type="button"
        class="quiet icon"
        aria-label="Fit to window"
        title="Fit to window"
        aria-pressed={fitted}
        disabled={!zoomable}
        onclick={fitToWindow}
      >
        <Scan size={16} />
      </button>
      <button
        type="button"
        class="quiet icon"
        aria-label="Actual size"
        title="Actual size"
        disabled={!zoomable}
        onclick={actualSize}
      >
        <Expand size={16} />
      </button>
      {#if canFullscreen}
        <button
          type="button"
          class="quiet icon"
          aria-label={sheetFull ? 'Exit full screen' : 'Full screen'}
          title={sheetFull ? 'Exit full screen' : 'Full screen'}
          onclick={toggleFullscreen}
        >
          {#if sheetFull}<Minimize size={16} />{:else}<Maximize
              size={16}
            />{/if}
        </button>
      {/if}
      {@render actions?.(file)}
      <button
        type="button"
        class="quiet icon"
        aria-label="Close"
        title="Close"
        onclick={close}
      >
        <X size={16} />
      </button>
    </div>
    <div class="viewer-body" class:has-aside={!!aside}>
      <!-- The pointer pans and swipes here; every one of its gestures is
           also a button in the bar above -->
      <div
        class="viewer-stage"
        class:zoomed={!fitted}
        bind:this={stage}
        role="presentation"
        {onpointerdown}
        {onpointermove}
        {onpointerup}
        onpointercancel={() => (press = null)}
        {ondblclick}
      >
        <div
          class="viewer-frame"
          style:transform={fitted
            ? undefined
            : `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`}
        >
          {#key file.name}<MediaBody {file} />{/key}
        </div>
      </div>
      {#if aside}
        <aside class="viewer-aside">{@render aside(file)}</aside>
      {/if}
    </div>
  </Modal>
{/if}
