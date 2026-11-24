import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { createRawSnippet } from 'svelte'
import MediaViewer from './MediaViewer.svelte'
import { holdLayer } from './layers.svelte'
import type { MediaFile } from '../media'

vi.mock('../api', () => ({
  api: {
    listWorkspaces: vi.fn(() => new Promise(() => {})),
    fileText: vi.fn(() => Promise.resolve('a text file')),
  },
}))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

// The stylesheet's text: vitest stubs a stylesheet's import, so it is read
// from disk (tests run from ui/), as OutputTile.test.ts does
const FS = 'node:fs'
const { readFileSync } = (await import(/* @vite-ignore */ FS)) as {
  readFileSync: (path: string, encoding: 'utf8') => string
}

const file = (name: string, kind: MediaFile['kind']): MediaFile => ({
  name,
  kind,
  url: `/outputs/${name}`,
  label: name.split('/').pop()!,
  workspace: 'studio',
})
const items = [
  file('run/a.png', 'image'),
  file('run/b.mp4', 'video'),
  file('run/c.png', 'image'),
]

const flush = () => new Promise((r) => setTimeout(r, 0))
const hashchange = (change: () => void) =>
  new Promise<void>((resolve) => {
    const on = () => {
      window.removeEventListener('hashchange', on)
      void flush().then(() => resolve())
    }
    window.addEventListener('hashchange', on)
    change()
  })

function viewer(props: Record<string, unknown> = {}) {
  return render(MediaViewer, { props: { items, open: 'run/b.mp4', ...props } })
}
const dialog = () => screen.findByRole('dialog', { name: /\.(png|mp4)$/ })
const press = (target: Element, init: KeyboardEventInit) =>
  fireEvent.keyDown(target, init)

beforeEach(() => {
  // the page was loaded on the link, so nothing is pushed on mount
  history.replaceState(
    history.state,
    '',
    '#/ws/studio/gallery?open=run%2Fb.mp4',
  )
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

it('opens on the named file and says where it is', async () => {
  viewer()
  const sheet = await screen.findByRole('dialog', { name: 'b.mp4' })
  expect(sheet).toHaveTextContent('2 of 3')
  expect(sheet.querySelector('video')).not.toBeNull()
  await waitFor(() => expect(sheet.contains(document.activeElement)).toBe(true))
  for (const name of [
    'Previous file',
    'Next file',
    'Zoom in',
    'Zoom out',
    'Fit to window',
    'Actual size',
    'Close',
  ])
    expect(screen.getByRole('button', { name })).toBeInTheDocument()
})

it('Next and Previous walk the items, and stop at the ends', async () => {
  viewer()
  await dialog()
  await fireEvent.click(screen.getByRole('button', { name: 'Next file' }))
  expect(
    await screen.findByRole('dialog', { name: 'c.png' }),
  ).toHaveTextContent('3 of 3')
  expect(screen.getByRole('button', { name: 'Next file' })).toHaveAttribute(
    'aria-disabled',
    'true',
  )
  // a step replaces the entry's open=
  expect(location.hash).toBe('#/ws/studio/gallery?open=run%2Fc.png')
  await fireEvent.click(screen.getByRole('button', { name: 'Previous file' }))
  await fireEvent.click(screen.getByRole('button', { name: 'Previous file' }))
  expect(
    await screen.findByRole('dialog', { name: 'a.png' }),
  ).toHaveTextContent('1 of 3')
  expect(screen.getByRole('button', { name: 'Previous file' })).toHaveAttribute(
    'aria-disabled',
    'true',
  )
})

it('stepping to an end with the button keeps the arrows working', async () => {
  viewer()
  await dialog()
  const next = screen.getByRole('button', { name: 'Next file' })
  next.focus()
  await fireEvent.click(next)
  await screen.findByRole('dialog', { name: 'c.png' })
  // at the end the button keeps its focus, inside the sheet's key capture
  expect(next).not.toBeDisabled()
  expect(document.activeElement).toBe(next)
  // a click on it at the end does nothing
  await fireEvent.click(next)
  expect(screen.getByRole('dialog', { name: 'c.png' })).toBeTruthy()
  await press(document.activeElement!, { key: 'ArrowLeft' })
  expect(await screen.findByRole('dialog', { name: 'b.mp4' })).toBeTruthy()
  expect(await dialog()).toContainElement(document.activeElement as HTMLElement)
})

it('a file that left the items steps on from where it stood', async () => {
  const { rerender } = viewer()
  await dialog()
  // b is unstarred with Starred on: the list no longer holds it
  await rerender({ items: [items[0], items[2]] })
  await fireEvent.click(screen.getByRole('button', { name: 'Next file' }))
  expect(await screen.findByRole('dialog', { name: 'c.png' })).toBeTruthy()
})

it('arrows step after a click on the video', async () => {
  viewer()
  const sheet = await dialog()
  const video = sheet.querySelector('video')!
  expect(video).toHaveAttribute('controls')
  expect(video.getAttribute('tabindex')).toBe('-1')
  await fireEvent.click(video)
  video.focus()
  const right = new KeyboardEvent('keydown', {
    key: 'ArrowRight',
    bubbles: true,
    cancelable: true,
  })
  video.dispatchEvent(right)
  // handled, so the focused video never seeks on it
  expect(right.defaultPrevented).toBe(true)
  expect(await screen.findByRole('dialog', { name: 'c.png' })).toBeTruthy()
  await press(document.activeElement ?? sheet, { key: 'ArrowLeft' })
  expect(await screen.findByRole('dialog', { name: 'b.mp4' })).toBeTruthy()
})

it('Shift-arrows seek the video 5 s', async () => {
  viewer()
  const video = (await dialog()).querySelector('video')!
  video.currentTime = 10
  await press(video, { key: 'ArrowRight', shiftKey: true })
  expect(video.currentTime).toBe(15)
  await press(video, { key: 'ArrowLeft', shiftKey: true })
  expect(video.currentTime).toBe(10)
  expect(screen.getByRole('dialog', { name: 'b.mp4' })).toBeTruthy()
})

it('keys typed in a field or under a dialog do nothing', async () => {
  const onkey = vi.fn(() => true)
  viewer({
    onkey,
    aside: createRawSnippet(() => ({
      render: () => '<label>Note <input type="text" /></label>',
    })),
  })
  const sheet = await dialog()
  const field = sheet.querySelector('input')!
  await press(field, { key: 'ArrowRight' })
  await press(field, { key: 's' })
  expect(screen.getByRole('dialog', { name: 'b.mp4' })).toHaveTextContent(
    '2 of 3',
  )
  expect(onkey).not.toHaveBeenCalled()
  // a dialog above the viewer (a confirm an action asked) has the keys
  const release = holdLayer()
  await press(sheet, { key: 'ArrowRight' })
  await press(sheet, { key: 's' })
  release()
  expect(screen.getByRole('dialog', { name: 'b.mp4' })).toHaveTextContent(
    '2 of 3',
  )
  expect(onkey).not.toHaveBeenCalled()
})

it('a tall image fits with its controls visible', async () => {
  viewer({ open: 'run/a.png' })
  const sheet = await screen.findByRole('dialog', { name: 'a.png' })
  const img = sheet.querySelector('img')!
  const stage = img.closest('.viewer-stage')
  expect(stage).not.toBeNull()
  expect(img.classList).toContain('media-fit')
  // the sheet is the window, the stage takes what the bar leaves and the
  // picture is bounded by the stage: nothing scrolls at fit
  const patterns = readFileSync('src/patterns.css', 'utf8')
  const rule = (name: string) =>
    patterns.match(new RegExp(`\\${name}\\s*\\{([^}]*)\\}`))?.[1] ?? ''
  expect(rule('.ui-sheet-fill')).toMatch(/inset:\s*0/)
  expect(rule('.ui-sheet-fill')).toMatch(/overflow:\s*hidden/)
  expect(rule('.viewer-stage')).toMatch(/min-height:\s*0/)
  expect(rule('.viewer-stage')).toMatch(/overflow:\s*hidden/)
  expect(rule('.media-fit')).toMatch(/max-height:\s*100%/)
  expect(rule('.media-fit')).toMatch(/object-fit:\s*contain/)
  // Close is in the bar above the stage, so it is on the first screen
  const close = screen.getByRole('button', { name: 'Close' })
  expect(
    close.compareDocumentPosition(stage!) & Node.DOCUMENT_POSITION_FOLLOWING,
  ).toBeTruthy()
})

it("Escape closes, and the page's onkey sees S", async () => {
  const onkey = vi.fn((e: KeyboardEvent) => e.key === 's')
  const { component } = viewer({ onkey })
  const sheet = await dialog()
  const s = new KeyboardEvent('keydown', {
    key: 's',
    bubbles: true,
    cancelable: true,
  })
  sheet.dispatchEvent(s)
  expect(onkey).toHaveBeenCalledWith(s, items[1])
  expect(s.defaultPrevented).toBe(true)
  await press(sheet, { key: 'Escape' })
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  // a deep link: open= is replaced away
  expect(location.hash).toBe('#/ws/studio/gallery')
  void component
})

it('the Close button closes', async () => {
  viewer()
  await dialog()
  await fireEvent.click(screen.getByRole('button', { name: 'Close' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
})

it('Back closes the viewer it opened', async () => {
  history.replaceState(history.state, '', '#/ws/studio/gallery')
  const { rerender } = viewer({ open: null })
  expect(screen.queryByRole('dialog')).toBeNull()
  // the page opens a file: a history entry is pushed
  await hashchange(() => void rerender({ open: 'run/a.png' }))
  expect(location.hash).toBe('#/ws/studio/gallery?open=run%2Fa.png')
  await screen.findByRole('dialog', { name: 'a.png' })
  await hashchange(() => history.back())
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  expect(location.hash).toBe('#/ws/studio/gallery')
})

it('a name no item has renders nothing', async () => {
  const onkey = vi.fn(() => true)
  viewer({ open: 'run/gone.png', onkey })
  await flush()
  expect(screen.queryByRole('dialog')).toBeNull()
  await press(document.body, { key: 's' })
  expect(onkey).not.toHaveBeenCalled()
})

it('focus goes into the viewer and back to the tile that opened it', async () => {
  history.replaceState(history.state, '', '#/ws/studio/gallery')
  const tile = document.createElement('button')
  tile.textContent = 'a.png'
  document.body.append(tile)
  tile.focus()
  const { rerender } = viewer({ open: null })
  await hashchange(() => void rerender({ open: 'run/a.png' }))
  const sheet = await screen.findByRole('dialog', { name: 'a.png' })
  await waitFor(() => expect(sheet.contains(document.activeElement)).toBe(true))
  // Close goes back: wait for it, so no later test sees it land
  await hashchange(
    () => void fireEvent.click(screen.getByRole('button', { name: 'Close' })),
  )
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  await waitFor(() => expect(document.activeElement).toBe(tile))
  tile.remove()
})

it('a text file shows its contents, audio its native controls', async () => {
  const listed = [file('run/notes.txt', 'text'), file('run/tone.wav', 'audio')]
  viewer({ items: listed, open: 'run/notes.txt' })
  const sheet = await screen.findByRole('dialog', { name: 'notes.txt' })
  await waitFor(() => expect(sheet).toHaveTextContent('a text file'))
  await fireEvent.click(screen.getByRole('button', { name: 'Next file' }))
  const audio = (
    await screen.findByRole('dialog', { name: 'tone.wav' })
  ).querySelector('audio')!
  expect(audio).toHaveAttribute('controls')
  expect(screen.getByRole('button', { name: 'Zoom in' })).toBeDisabled()
})

// A quick flat stroke: a swipe when it lands on the stage or the picture
async function stroke(target: Element) {
  await fireEvent.pointerDown(target, { clientX: 300, clientY: 100 })
  await fireEvent.pointerUp(target, { clientX: 100, clientY: 100 })
  await flush()
}
const position = async () => (await dialog()).textContent?.match(/\d of 3/)?.[0]

it('a swipe on the picture or the stage steps', async () => {
  // an earlier test's Back lands first; then opened on its link
  await flush()
  history.replaceState({}, '', '#/ws/studio/gallery?open=run%2Fa.png')
  viewer({ open: 'run/a.png' })
  const sheet = await screen.findByRole('dialog', { name: 'a.png' })
  await stroke(sheet.querySelector('img')!)
  expect(await screen.findByRole('dialog', { name: 'b.mp4' })).toBeTruthy()
  await stroke((await dialog()).querySelector('.viewer-stage')!)
  expect(await screen.findByRole('dialog', { name: 'c.png' })).toBeTruthy()
})

it("a drag on a clip's controls, audio, text or a control never swipes", async () => {
  const listed = [
    file('run/a.png', 'image'),
    file('run/b.mp4', 'video'),
    file('run/tone.wav', 'audio'),
    file('run/notes.txt', 'text'),
  ]
  for (const [open, pick] of [
    ['run/b.mp4', 'video'],
    ['run/tone.wav', 'audio'],
    ['run/notes.txt', 'pre'],
    ['run/b.mp4', 'button[aria-label="Zoom in"]'],
  ] as const) {
    const { unmount } = viewer({ items: listed, open })
    const anyDialog = () => screen.findByRole('dialog')
    const sheet = await anyDialog()
    const at = sheet.textContent?.match(/\d of 4/)?.[0]
    await waitFor(() => expect(sheet.querySelector(pick)).not.toBeNull())
    await stroke(sheet.querySelector(pick)!)
    expect((await anyDialog()).textContent?.match(/\d of 4/)?.[0]).toBe(at)
    unmount()
  }
})

it('a drag on a zoomed clip scrubs, it does not pan', async () => {
  viewer()
  const sheet = await dialog()
  const stage = sheet.querySelector<HTMLElement>('.viewer-stage')!
  const video = sheet.querySelector('video')!
  // jsdom lays nothing out: a 400 x 300 stage the clip fills at fit
  const size = (el: Element, w: number, h: number, as: 'client' | 'offset') => {
    Object.defineProperty(el, `${as}Width`, { configurable: true, value: w })
    Object.defineProperty(el, `${as}Height`, { configurable: true, value: h })
  }
  size(stage, 400, 300, 'client')
  size(video, 400, 300, 'offset')
  await fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }))
  const frame = sheet.querySelector<HTMLElement>('.viewer-frame')!
  await waitFor(() => expect(frame.style.transform).toMatch(/scale/))
  const before = frame.style.transform
  const drag = async (on: Element) => {
    await fireEvent.pointerDown(on, { clientX: 300, clientY: 100 })
    await fireEvent.pointerMove(on, { clientX: 250, clientY: 100 })
    await fireEvent.pointerUp(on, { clientX: 250, clientY: 100 })
  }
  await drag(video)
  expect(frame.style.transform).toBe(before)
  expect(await position()).toBe('2 of 3')
  // the stage's own margin still pans
  await drag(stage)
  expect(frame.style.transform).not.toBe(before)
})
