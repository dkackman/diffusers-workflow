// The media viewer's rules (ui/MediaViewer.svelte): how far it zooms, what a
// swipe and a key mean, how far a zoomed picture pans - and the open file in
// the hash, as Back expects it.
//
// A page keeps the open file as `open=<name>` in the hash's query. The page
// seeds its `open` from `openInHash()` and binds it to the viewer, which
// writes every change through `writeOpen` (Don, 2026-11-23: Back closes the
// viewer):
// - opening a file PUSHES an entry, so the router takes it as one
//   navigation on the same path - the page does not move and its heading
//   does not take focus; the viewer does
// - previous and next REPLACE the entry's open=, so Back never walks every
//   file that was looked at
// - closing goes BACK when the viewer pushed the entry, so the next Back
//   leaves the page; a page loaded on the link (no pushed entry) has open=
//   replaced away
// - Back while the viewer is open is a hashchange with no open=: the viewer
//   reads `openInHash()` and closes
import { hashQuery, queryHash, setHashQuery } from './router.svelte'

export const ZOOM_MAX = 8
/** Each zoom step's factor */
const ZOOM_STEP = 1.5
/** How far a seek key moves a clip, in seconds */
export const SEEK_SECONDS = 5

/** The zoom one step in (`by` 1) or out (-1) from `scale`: the steps are
 * `fitScale` times powers of 1.5, up to ZOOM_MAX and never below
 * `fitScale` (a picture smaller than the window fits above 1). */
export function nextZoom(scale: number, fitScale: number, by: 1 | -1): number {
  const at = Math.log(scale / fitScale) / Math.log(ZOOM_STEP)
  const eps = 1e-6
  const level = by === 1 ? Math.floor(at + eps) + 1 : Math.ceil(at - eps) - 1
  const next = fitScale * ZOOM_STEP ** level
  return Math.max(fitScale, Math.min(Math.max(ZOOM_MAX, fitScale), next))
}

/** A pointer's travel as a step: a quick, flat stroke of 50 px or more;
 * a swipe to the left is the next file. */
export function swipeStep(dx: number, dy: number, ms: number): 1 | -1 | 0 {
  if (Math.abs(dx) < 50 || Math.abs(dx) <= 2 * Math.abs(dy) || ms > 600)
    return 0
  return dx < 0 ? 1 : -1
}

export type ViewerKey =
  | 'prev'
  | 'next'
  | 'zoomIn'
  | 'zoomOut'
  | 'fit'
  | 'actual'
  | 'fullscreen'
  | 'seekBack'
  | 'seekOn'

const KEYS: Record<string, ViewerKey> = {
  '+': 'zoomIn',
  '=': 'zoomIn',
  '-': 'zoomOut',
  '0': 'fit',
  '1': 'actual',
  f: 'fullscreen',
}

/** What a key does in the viewer, or null when it is not the viewer's: a
 * key held with Ctrl, Alt or Meta never is (the browser's and the page's
 * own shortcuts). Shift turns an arrow from a step into a seek. */
export function viewerKey(e: KeyboardEvent): ViewerKey | null {
  if (e.ctrlKey || e.altKey || e.metaKey) return null
  if (e.key === 'ArrowLeft') return e.shiftKey ? 'seekBack' : 'prev'
  if (e.key === 'ArrowRight') return e.shiftKey ? 'seekOn' : 'next'
  return KEYS[e.key] ?? null
}

type Size = { w: number; h: number }

/** A pan (the picture's centre offset from the box's, in px) held so no
 * edge of the picture comes inside the box; a picture no larger than the
 * box stays centred on that axis. */
export function clampPan(
  pan: { x: number; y: number },
  scale: number,
  box: Size,
  media: Size,
): { x: number; y: number } {
  const room = (length: number, inside: number) =>
    Math.max(0, (length * scale - inside) / 2)
  const hold = (value: number, limit: number) =>
    Math.max(-limit, Math.min(limit, value)) || 0
  return {
    x: hold(pan.x, room(media.w, box.w)),
    y: hold(pan.y, room(media.h, box.h)),
  }
}

/** The file the hash's query names as open, or null. */
export function openInHash(): string | null {
  return hashQuery().open || null
}

// The mark on an entry the viewer pushed, kept in history.state beside the
// router's own index (the router keeps what it finds there)
const PUSHED = 'dwViewerPushed'

function pushedHere(): boolean {
  const state = history.state as Record<string, unknown> | null
  return state?.[PUSHED] === true
}

function markPushed() {
  const state = history.state as Record<string, unknown> | null
  history.replaceState({ ...(state ?? {}), [PUSHED]: true }, '')
}

/** Write the open file into the hash the way Back expects (the head of
 * this file): open pushes, a step replaces, close goes back or, on a
 * deep link, replaces open= away. The same file again writes nothing. */
export function writeOpen(name: string | null) {
  const query = hashQuery()
  const now = query.open || null
  if (name === now) return
  if (name && !now) {
    // after the router's own listener has stamped the new entry
    window.addEventListener('hashchange', markPushed, { once: true })
    location.hash = queryHash({ ...query, open: name })
  } else if (name) setHashQuery({ ...query, open: name })
  else if (pushedHere()) history.back()
  else setHashQuery({ ...query, open: '' })
}
