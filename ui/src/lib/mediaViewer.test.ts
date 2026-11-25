import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import {
  ZOOM_MAX,
  clampPan,
  nextZoom,
  openInHash,
  pinchZoom,
  swipeStep,
  viewerKey,
  writeOpen,
} from './mediaViewer'
import { placeScroll } from './router.svelte'
import { api } from './test/apiMock'

vi.mock('./api', async () => (await import('./test/apiMock')).apiModule())
api.listWorkspaces.mockImplementation(() => new Promise(() => {}))
vi.mock('./toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

const key = (init: KeyboardEventInit) => new KeyboardEvent('keydown', init)

it('zoom steps from fit and stops at 8', () => {
  expect(ZOOM_MAX).toBe(8)
  expect(nextZoom(0.5, 0.5, 1)).toBeCloseTo(0.75)
  expect(nextZoom(0.75, 0.5, 1)).toBeCloseTo(1.125)
  let scale = 0.5
  for (let i = 0; i < 20; i++) scale = nextZoom(scale, 0.5, 1)
  expect(scale).toBe(8)
  // from the cap, out is the last step below it
  expect(nextZoom(8, 1, -1)).toBeCloseTo(1.5 ** 5)
})

it('never zooms out past fit', () => {
  expect(nextZoom(0.75, 0.5, -1)).toBeCloseTo(0.5)
  expect(nextZoom(0.5, 0.5, -1)).toBe(0.5)
  // a picture smaller than the window at actual size: fit is above 1
  expect(nextZoom(1, 2, -1)).toBe(2)
})

it('a short flat swipe steps, a slow or steep one does not', () => {
  expect(swipeStep(-80, 10, 200)).toBe(1)
  expect(swipeStep(80, -10, 200)).toBe(-1)
  expect(swipeStep(-40, 0, 200)).toBe(0)
  expect(swipeStep(-80, 50, 200)).toBe(0)
  expect(swipeStep(-80, 10, 900)).toBe(0)
})

it('the arrows step and Shift-arrows seek', () => {
  expect(viewerKey(key({ key: 'ArrowLeft' }))).toBe('prev')
  expect(viewerKey(key({ key: 'ArrowRight' }))).toBe('next')
  expect(viewerKey(key({ key: 'ArrowLeft', shiftKey: true }))).toBe('seekBack')
  expect(viewerKey(key({ key: 'ArrowRight', shiftKey: true }))).toBe('seekOn')
  expect(viewerKey(key({ key: '+', shiftKey: true }))).toBe('zoomIn')
  expect(viewerKey(key({ key: '=' }))).toBe('zoomIn')
  expect(viewerKey(key({ key: '-' }))).toBe('zoomOut')
  expect(viewerKey(key({ key: '0' }))).toBe('fit')
  expect(viewerKey(key({ key: '1' }))).toBe('actual')
  expect(viewerKey(key({ key: 'f' }))).toBe('fullscreen')
  expect(viewerKey(key({ key: 'Escape' }))).toBeNull()
  // a modifier other than Shift is never the viewer's
  expect(viewerKey(key({ key: 'ArrowLeft', metaKey: true }))).toBeNull()
  expect(viewerKey(key({ key: 'ArrowRight', altKey: true }))).toBeNull()
  expect(viewerKey(key({ key: 'f', ctrlKey: true }))).toBeNull()
})

it("a pan is clamped to the picture's edges", () => {
  const box = { w: 1000, h: 800 }
  const media = { w: 500, h: 400 }
  // at 4x the picture is 2000 x 1600: 500 and 400 to spare each side
  expect(clampPan({ x: 900, y: -900 }, 4, box, media)).toEqual({
    x: 500,
    y: -400,
  })
  expect(clampPan({ x: 100, y: 50 }, 4, box, media)).toEqual({ x: 100, y: 50 })
  // smaller than the box: it stays centred
  expect(clampPan({ x: 100, y: 50 }, 1, box, media)).toEqual({ x: 0, y: 0 })
})

// The open file in the hash, as Back expects it (Don, 2026-11-23): opening
// pushes, a step replaces, closing goes back to the entry before

const flush = () => new Promise((r) => setTimeout(r, 0))
function hashchange(change: () => void) {
  return new Promise<void>((resolve) => {
    const on = () => {
      window.removeEventListener('hashchange', on)
      void flush().then(() => resolve())
    }
    window.addEventListener('hashchange', on)
    change()
  })
}
const go = (hash: string) => hashchange(() => (location.hash = hash))

function scrolled(y: number) {
  Object.defineProperty(window, 'scrollY', { value: y, configurable: true })
}

beforeEach(() => {
  // in place, with no mark an earlier test's entry carried
  const { dwIndex } = (history.state ?? {}) as { dwIndex?: number }
  history.replaceState({ dwIndex }, '', '#/ws/studio/gallery?q=cat')
})
afterEach(() => {
  vi.restoreAllMocks()
  scrolled(0)
})

it('opening a file pushes an entry, keeping the query', async () => {
  const before = history.length
  await hashchange(() => writeOpen('a/one.png'))
  expect(history.length).toBe(before + 1)
  expect(openInHash()).toBe('a/one.png')
  expect(location.hash).toBe('#/ws/studio/gallery?q=cat&open=a%2Fone.png')
})

it('previous and next replace the entry', async () => {
  await hashchange(() => writeOpen('a/one.png'))
  const before = history.length
  writeOpen('a/two.png')
  expect(history.length).toBe(before)
  expect(openInHash()).toBe('a/two.png')
  // Back leaves the viewer, not the file before
  await hashchange(() => history.back())
  expect(openInHash()).toBeNull()
  expect(location.hash).toBe('#/ws/studio/gallery?q=cat')
})

it('closing goes back to the entry before the file opened', async () => {
  await hashchange(() => writeOpen('a/one.png'))
  writeOpen('a/two.png')
  const before = history.length
  await hashchange(() => writeOpen(null))
  expect(openInHash()).toBeNull()
  expect(location.hash).toBe('#/ws/studio/gallery?q=cat')
  // a traversal, not a new entry: Forward would reopen the file
  expect(history.length).toBe(before)
})

it('a deep link closes by replacing open= away', async () => {
  // the page was loaded on the link: no entry the viewer pushed
  await go('#/ws/studio/gallery?q=cat&open=a%2Fone.png')
  const before = history.length
  const back = vi.spyOn(history, 'back')
  writeOpen(null)
  expect(back).not.toHaveBeenCalled()
  expect(history.length).toBe(before)
  expect(location.hash).toBe('#/ws/studio/gallery?q=cat')
})

it('the same file again writes nothing', async () => {
  await hashchange(() => writeOpen('a/one.png'))
  const before = history.length
  const hash = location.hash
  writeOpen('a/one.png')
  expect(history.length).toBe(before)
  expect(location.hash).toBe(hash)
})

it('opening and closing never move the page', async () => {
  const scrollTo = vi
    .spyOn(window, 'scrollTo')
    .mockImplementation((_x?: unknown, y?: unknown) => scrolled(Number(y)))
  Object.defineProperty(document.documentElement, 'scrollHeight', {
    value: innerHeight + 5000,
    configurable: true,
  })
  placeScroll()
  scrollTo.mockClear()
  scrolled(1200)
  await hashchange(() => writeOpen('a/one.png'))
  placeScroll()
  expect(scrollTo).not.toHaveBeenCalled()
  writeOpen('a/two.png')
  placeScroll()
  expect(scrollTo).not.toHaveBeenCalled()
  await hashchange(() => writeOpen(null))
  placeScroll()
  // Back lands where the page was left: the same place
  for (const [, y] of scrollTo.mock.calls) expect(y).toBe(1200)
  expect(window.scrollY).toBe(1200)
})

it('pinchZoom scales the zoom by how far the fingers spread, held to 1..8', () => {
  const at = (
    a: number,
    b: number,
  ): [{ x: number; y: number }, { x: number; y: number }] => [
    { x: a, y: 0 },
    { x: b, y: 0 },
  ]
  expect(pinchZoom(at(0, 100), at(0, 200), 1)).toBe(2)
  expect(pinchZoom(at(0, 100), at(0, 200), 3)).toBe(6)
  expect(pinchZoom(at(0, 100), at(0, 50), 1)).toBe(1)
  expect(pinchZoom(at(0, 100), at(0, 5000), 2)).toBe(ZOOM_MAX)
  // both fingers on one point: nothing to measure
  expect(pinchZoom(at(5, 5), at(0, 200), 2)).toBe(2)
})
