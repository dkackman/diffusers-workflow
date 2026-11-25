<script lang="ts">
  // One file, large, fitted to the window: zoom to 8x (buttons, keys, a
  // pinch) with a drag to pan, previous and next through the page's own
  // list, a swipe at fit, full screen. Every key has a button. A page keeps the open file in the hash as open= and
  // binds `open`; what opening, stepping, closing and Back do to the
  // history is mediaViewer.ts's (its head says why). An `open` no item
  // names renders nothing - the page says the file is gone - but a file
  // that leaves `items` while it is shown (deleted, filtered, unstarred
  // under Starred) steps on to whatever now fills its place, or the one
  // before it; with nothing left the viewer closes. The look is `.ui-sheet-fill` and `.viewer-*` in patterns.css.
  import { tick, untrack, type Snippet } from 'svelte'
  import {
    ChevronLeft,
    ChevronRight,
    Ellipsis,
    Expand,
    Shrink,
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
    pinchTracker,
    openInHash,
    swipeStep,
    viewerKey,
    writeOpen,
    type ViewerKey,
  } from '../mediaViewer'
  import { leafName } from '../names'
  import { neighbor } from '../neighbor'
  import { workspace } from '../workspaceState.svelte'
  import { overlayOpen } from './layers.svelte'
  import MediaBody from './MediaBody.svelte'
  import Modal from './Modal.svelte'
  import Popover from './Popover.svelte'

  let {
    items,
    open = $bindable(null),
    actions,
    aside,
    onkey,
    ongone,
  }: {
    items: MediaFile[]
    open: string | null
    /** The page's own buttons for the file, before Close; in the More menu
     * at a phone's width, where an icon button shows its aria-label */
    actions?: Snippet<[MediaFile]>
    /** What the page says about the file, beside the stage */
    aside?: Snippet<[MediaFile]>
    /** A key the viewer does not take; true when the page handled it */
    onkey?: (e: KeyboardEvent, file: MediaFile) => boolean
    /** The file's own read came back 404 (deleted while listed): the page
     * says it is missing (missingFile's `gone`) */
    ongone?: (name: string) => void
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

  // The shown file left the list: on from where it stood (the head)
  $effect(() => {
    const gone = file && listedAt < 0 ? file.name : null
    if (gone === null) return
    untrack(() => {
      open =
        neighbor(names, gone, 1, lastAt) ?? neighbor(names, gone, -1, lastAt)
    })
  })

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
  // Another workspace's files are not these: a switch closes it
  let shownIn = untrack(() => workspace.current)
  $effect(() => {
    const ws = workspace.current
    untrack(() => {
      if (ws !== shownIn) [shownIn, open] = [ws, null]
    })
  })

  // A file that fails to draw is asked for again: a 404 is one the listing
  // still names and the server no longer holds
  async function onerrorcapture() {
    if (!ongone || !file) return
    const name = file.name
    const res = await fetch(file.url, { method: 'HEAD' }).catch(() => null)
    if (res?.status === 404 && open === name) ongone(name)
  }
  // A button in the bar that cannot act (here, and in a page's actions) is
  // aria-disabled with a press that does nothing, never disabled: a
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
  const fullLabel = $derived(sheetFull ? 'Exit full screen' : 'Full screen')
  // One toggle for the two views a double-click also swaps between
  const sizeLabel = $derived(fitted ? 'Actual size' : 'Fit to window')

  // At a phone's width the bar keeps one row: zoom and full screen go (a
  // pinch or double-tap zooms) and the page's actions go into More. Read
  // off the window, as the stylesheet's own break is
  const NARROW_PX = 600
  const isNarrow = () =>
    typeof window !== 'undefined' && window.innerWidth <= NARROW_PX
  let narrow = $state(isNarrow())
  $effect(() => {
    const sync = () => (narrow = isNarrow())
    window.addEventListener('resize', sync)
    return () => window.removeEventListener('resize', sync)
  })
  let moreOpen = $state(false)
  let moreButton = $state<HTMLButtonElement | null>(null)
  let moreList = $state<HTMLElement | null>(null)
  // Any action chosen from More closes it first, so a question or a dialog
  // it asks is not under the menu. A listener, not an attribute: the list
  // is the page's buttons, and the press is theirs
  $effect(() => {
    const list = moreList
    if (!list) return
    const chosen = (e: Event) => {
      if (!(e.target instanceof Element)) return
      const pressed = e.target.closest('button, a')
      if (pressed && pressed.getAttribute('aria-disabled') !== 'true')
        moreOpen = false
    }
    list.addEventListener('click', chosen)
    return () => list.removeEventListener('click', chosen)
  })
  $effect(() => {
    if (!narrow || !file) moreOpen = false
  })
  // When More shuts, or a dialog from it ends, focus left on <body> or in
  // the menu goes to More, or to Close once the trigger has unmounted
  let closeButton = $state<HTMLButtonElement | null>(null)
  let [wasMore, wasOverlay] = [false, false]
  $effect(() => {
    const more = moreOpen
    const above = file ? overlayOpen(1) : false
    untrack(() => {
      const ended = (wasMore && !more) || wasOverlay
      ;[wasMore, wasOverlay] = [more, above]
      if (ended && !above) void settleFocus()
    })
  })
  async function settleFocus() {
    await tick()
    const at = document.activeElement
    if (at && at !== document.body && !moreList?.contains(at)) return
    ;(moreButton?.isConnected ? moreButton : closeButton)?.focus()
  }

  const ACTS: Record<ViewerKey, () => void> = {
    prev: () => step(prevName),
    next: () => step(nextName),
    zoomIn: () => zoomable && zoomBy(1),
    zoomOut: () => zoomable && zoomBy(-1),
    fit: fitToWindow,
    actual: () => zoomable && actualSize(),
    // At a phone's width the bar hides full screen, so the key only leaves it
    fullscreen: () => (!narrow || sheetFull) && toggleFullscreen(),
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

  // A drag pans when zoomed; two fingers pinch; a quick flat stroke at fit
  // steps. Only a press on the stage, its frame or a picture: a clip's seek
  // bar, text or a control never starts one
  let press: { x: number; y: number; t: number; pan: typeof pan } | null = null
  const fingers = pinchTracker()
  function gestureTarget(target: EventTarget | null): boolean {
    if (!(target instanceof Element)) return false
    if (target === stage || target instanceof HTMLImageElement) return true
    return target.classList.contains('viewer-frame')
  }
  function onpointerdown(e: PointerEvent) {
    if (!gestureTarget(e.target)) return void (press = null)
    // The stage's until it lifts, wherever it travels
    stage?.setPointerCapture?.(e.pointerId)
    window.addEventListener('pointerup', onpointerup)
    window.addEventListener('pointercancel', onpointerup)
    const at = { x: e.clientX, y: e.clientY }
    const pinching = zoomable && fingers.press(e.pointerId, at, zoom)
    press = pinching ? null : { ...at, t: e.timeStamp, pan }
  }
  function onpointermove(e: PointerEvent) {
    const pinched = fingers.move(e.pointerId, { x: e.clientX, y: e.clientY })
    if (pinched !== null) return setZoom(pinched)
    if (!press || fitted) return
    pan = heldPan({
      x: press.pan.x + e.clientX - press.x,
      y: press.pan.y + e.clientY - press.y,
    })
  }
  function onpointerup(e: PointerEvent) {
    const pinched = fingers.lift(e.pointerId)
    if (fingers.size === 0) {
      window.removeEventListener('pointerup', onpointerup)
      window.removeEventListener('pointercancel', onpointerup)
    }
    const was = press
    press = null
    if (!was || pinched || e.type === 'pointercancel' || !fitted) return
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
    size="fill"
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
      {#if !narrow}
        <button
          type="button"
          class="quiet icon"
          aria-label="Zoom out"
          title="Zoom out"
          aria-disabled={!zoomable || fitted}
          onclick={ACTS.zoomOut}
        >
          <ZoomOut size={16} />
        </button>
        <button
          type="button"
          class="quiet icon"
          aria-label="Zoom in"
          title="Zoom in"
          aria-disabled={!zoomable}
          onclick={ACTS.zoomIn}
        >
          <ZoomIn size={16} />
        </button>
        <button
          type="button"
          class="quiet viewer-size"
          aria-label={sizeLabel}
          title={sizeLabel}
          aria-disabled={!zoomable}
          onclick={fitted ? ACTS.actual : fitToWindow}
        >
          {fitted ? '100%' : 'Fit'}
        </button>
        {#if canFullscreen}
          <button
            type="button"
            class="quiet icon"
            aria-label={fullLabel}
            title={fullLabel}
            onclick={toggleFullscreen}
          >
            {#if sheetFull}<Shrink size={16} />{:else}<Expand size={16} />{/if}
          </button>
        {/if}
        {@render actions?.(file)}
      {:else if actions}
        <button
          type="button"
          class="quiet icon"
          bind:this={moreButton}
          aria-label="More"
          title="More"
          aria-haspopup="dialog"
          aria-expanded={moreOpen}
          onclick={() => (moreOpen = !moreOpen)}
        >
          <Ellipsis size={16} />
        </button>
      {/if}
      <button
        type="button"
        class="quiet icon"
        aria-label="Close"
        title="Close"
        bind:this={closeButton}
        onclick={close}
      >
        <X size={16} />
      </button>
    </div>
    {#if narrow && actions}
      <Popover
        bind:open={moreOpen}
        label="More"
        anchor={moreButton}
        align="end"
        menu
      >
        <div class="viewer-more" bind:this={moreList}>
          {@render actions?.(file)}
        </div>
      </Popover>
    {/if}
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
        {onerrorcapture}
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
