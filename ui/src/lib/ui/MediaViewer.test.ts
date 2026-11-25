import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { createRawSnippet } from 'svelte'
import MediaViewer from './MediaViewer.svelte'
import { holdLayer } from './layers.svelte'
import type { MediaFile } from '../media'
import { api } from '../test/apiMock'
import { workspace } from '../workspaceState.svelte'

api.listWorkspaces.mockImplementation(() => new Promise(() => {}))
api.fileText.mockImplementation(() => Promise.resolve('a text file'))

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

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
  // the hash names studio; the router would set it on the first hashchange
  workspace.current = 'studio'
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
  // the viewer marks an entry it pushed; the next test starts unmarked
  history.replaceState(null, '', location.href)
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
  expect(await screen.findByRole('dialog', { name: 'c.png' })).toBeTruthy()
  expect(location.hash).toBe('#/ws/studio/gallery?open=run%2Fc.png')
  // the last one goes: back to the one before its place
  // (a rerender sets every prop again: `open` as a bound page holds it)
  await rerender({ items: [items[0]], open: 'run/c.png' })
  expect(await screen.findByRole('dialog', { name: 'a.png' })).toBeTruthy()
  // nothing left: the viewer closes
  await rerender({ items: [], open: 'run/a.png' })
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  expect(location.hash).toBe('#/ws/studio/gallery')
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

it('one button swaps between actual size and fit, saying which it does', async () => {
  viewer()
  await dialog()
  // fitted, it offers actual size; zoomed, it offers the fit back
  const toggle = screen.getByRole('button', { name: 'Actual size' })
  expect(toggle).toHaveTextContent('100%')
  expect(toggle).toHaveAttribute('title', 'Actual size')
  expect(
    screen.queryByRole('button', { name: 'Fit to window' }),
  ).not.toBeInTheDocument()
  await fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }))
  const fit = await screen.findByRole('button', { name: 'Fit to window' })
  expect(fit).toBe(toggle)
  expect(fit).toHaveTextContent('Fit')
  expect(fit).toHaveAttribute('title', 'Fit to window')
  await fireEvent.click(fit)
  expect(await screen.findByRole('button', { name: 'Actual size' })).toBe(
    toggle,
  )
})

const pageActions = createRawSnippet(() => ({
  render: () =>
    '<span class="acts"><button type="button" class="quiet icon" aria-label="Star">*</button></span>',
}))
const widthOf = (px: number) => {
  Object.defineProperty(window, 'innerWidth', { configurable: true, value: px })
  window.dispatchEvent(new Event('resize'))
}

it("at a phone's width the page's actions are in More, and zoom goes", async () => {
  widthOf(390)
  try {
    viewer({ actions: pageActions })
    const sheet = await dialog()
    const bar = sheet.querySelector('.viewer-bar')!
    // one row: the name, the place, previous and next, More and Close
    expect(bar.querySelector('[aria-label="Star"]')).toBeNull()
    for (const name of ['Zoom in', 'Zoom out', 'Actual size'])
      expect(screen.queryByRole('button', { name })).not.toBeInTheDocument()
    const more = screen.getByRole('button', { name: 'More' })
    expect(more).toHaveAttribute('aria-expanded', 'false')
    expect(more).toHaveAttribute('title', 'More')
    await fireEvent.click(more)
    const menu = await screen.findByRole('dialog', { name: 'More' })
    expect(more).toHaveAttribute('aria-expanded', 'true')
    const star = within(menu).getByRole('button', { name: 'Star' })
    // choosing an action closes the menu, so what it asks is not under it
    await fireEvent.click(star)
    await waitFor(() =>
      expect(
        screen.queryByRole('dialog', { name: 'More' }),
      ).not.toBeInTheDocument(),
    )
    // widened, the actions are back in the bar and More goes
    widthOf(1024)
    await waitFor(() =>
      expect(bar.querySelector('[aria-label="Star"]')).not.toBeNull(),
    )
    expect(screen.queryByRole('button', { name: 'More' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Zoom in' })).toBeInTheDocument()
  } finally {
    widthOf(1024)
  }
})

it("f enters full screen, but not at a phone's width, where the bar hides it", async () => {
  Object.defineProperty(document, 'fullscreenEnabled', {
    configurable: true,
    value: true,
  })
  const enter = vi.fn(() => Promise.resolve())
  Object.defineProperty(HTMLElement.prototype, 'requestFullscreen', {
    configurable: true,
    value: enter,
  })
  try {
    widthOf(390)
    viewer()
    const sheet = await dialog()
    await press(sheet, { key: 'f' })
    expect(enter).not.toHaveBeenCalled()
    widthOf(1024)
    await press(sheet, { key: 'f' })
    expect(enter).toHaveBeenCalledTimes(1)
  } finally {
    widthOf(1024)
    delete (document as { fullscreenEnabled?: boolean }).fullscreenEnabled
    delete (HTMLElement.prototype as { requestFullscreen?: unknown })
      .requestFullscreen
  }
})

it('a pan ends on a pointerup outside the viewer', async () => {
  history.replaceState({}, '', '#/ws/studio/gallery?open=run%2Fa.png')
  viewer({ open: 'run/a.png' })
  const sheet = await screen.findByRole('dialog', { name: 'a.png' })
  const stage = sheet.querySelector('.viewer-stage') as HTMLElement
  const capture = vi.fn()
  stage.setPointerCapture = capture
  await fireEvent.pointerDown(stage, {
    clientX: 300,
    clientY: 100,
    pointerId: 1,
  })
  expect(capture).toHaveBeenCalledWith(1)
  // released over the page behind: the gesture still ends, and steps
  await fireEvent.pointerUp(document.body, { clientX: 100, clientY: 100 })
  expect(await screen.findByRole('dialog', { name: 'b.mp4' })).toBeTruthy()
})

it('two pointers zoom by their spread', async () => {
  const sheet = await (viewer(), dialog())
  const stage = sheet.querySelector('.viewer-stage') as HTMLElement
  expect(stage).not.toHaveClass('zoomed')
  await fireEvent.pointerDown(stage, {
    clientX: 100,
    clientY: 100,
    pointerId: 1,
  })
  await fireEvent.pointerDown(stage, {
    clientX: 200,
    clientY: 100,
    pointerId: 2,
  })
  await fireEvent.pointerMove(stage, {
    clientX: 300,
    clientY: 100,
    pointerId: 2,
  })
  await waitFor(() => expect(stage).toHaveClass('zoomed'))
  // lifting both fingers is not a swipe
  await fireEvent.pointerUp(stage, { clientX: 300, clientY: 100, pointerId: 2 })
  await fireEvent.pointerUp(stage, { clientX: 100, clientY: 100, pointerId: 1 })
  expect(screen.getByRole('dialog', { name: 'b.mp4' })).toBeTruthy()
})

it('a workspace change closes the viewer', async () => {
  const was = workspace.current
  viewer()
  await dialog()
  workspace.current = 'elsewhere'
  try {
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  } finally {
    workspace.current = was
  }
})

it('stepping to a deleted neighbour says it is missing', async () => {
  const ongone = vi.fn()
  const fetched = vi
    .spyOn(globalThis, 'fetch')
    .mockResolvedValue(new Response(null, { status: 404 }))
  viewer({ ongone, open: 'run/a.png' })
  const sheet = await screen.findByRole('dialog', { name: 'a.png' })
  await fireEvent.error(sheet.querySelector('img')!)
  await waitFor(() => expect(ongone).toHaveBeenCalledWith('run/a.png'))
  expect(fetched).toHaveBeenCalled()
  // a file that merely fails to draw (not a 404) is not said gone
  ongone.mockClear()
  fetched.mockResolvedValue(new Response(null, { status: 200 }))
  await fireEvent.error(sheet.querySelector('img')!)
  await flush()
  expect(ongone).not.toHaveBeenCalled()
})

it('More closes on Escape and returns focus', async () => {
  widthOf(390)
  try {
    viewer({ actions: pageActions })
    await dialog()
    const more = screen.getByRole('button', { name: 'More' })
    await fireEvent.click(more)
    const menu = await screen.findByRole('dialog', { name: 'More' })
    await press(menu, { key: 'Escape' })
    await waitFor(() =>
      expect(screen.queryByRole('dialog', { name: 'More' })).toBeNull(),
    )
    // the viewer stays, and focus is back on More
    expect(screen.getByRole('dialog', { name: 'b.mp4' })).toBeTruthy()
    await waitFor(() => expect(document.activeElement).toBe(more))
  } finally {
    widthOf(1024)
  }
})

it('closing Use as input on a phone focuses the viewer', async () => {
  widthOf(390)
  try {
    viewer({ actions: pageActions })
    const sheet = await dialog()
    await fireEvent.click(screen.getByRole('button', { name: 'More' }))
    const menu = await screen.findByRole('dialog', { name: 'More' })
    // the action opens a dialog over the viewer; its trigger is gone
    const release = holdLayer()
    await fireEvent.click(within(menu).getByRole('button', { name: 'Star' }))
    ;(document.activeElement as HTMLElement | null)?.blur()
    release()
    await waitFor(() =>
      expect(sheet.contains(document.activeElement)).toBe(true),
    )
    expect(document.activeElement).not.toBe(document.body)
  } finally {
    widthOf(1024)
  }
})
