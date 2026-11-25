import { cleanup, fireEvent, render, waitFor } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import OutputTile from './OutputTile.svelte'
import { api } from '../test/apiMock'
import { assetMedia, type MediaFile } from '../media'
import { quiet, useQuietClock } from '../test/quiet'

// The stylesheet's text: vitest stubs a stylesheet's import, so it is read
// from disk (tests run from ui/). A specifier that is not a literal keeps
// node's types out of the app's program, as freshness.test.ts does
useQuietClock()

const FS = 'node:fs'
const { readFileSync } = (await import(/* @vite-ignore */ FS)) as {
  readFileSync: (path: string, encoding: 'utf8') => string
}

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
api.fileText.mockImplementation(() =>
  Promise.resolve('one\ntwo\nthree\nfour\n'),
)
const fileText = api.fileText

const media = (fields: Partial<MediaFile>): MediaFile => ({
  name: 'demo/a.png',
  kind: 'image',
  url: '/outputs/demo/a.png',
  label: 'a.png',
  workspace: 'default',
  thumb: '/thumb/demo/a.png',
  ...fields,
})

/** An IntersectionObserver the test drives: `show()` says every observed
 * element scrolled into view */
function stubObserver() {
  const seen: Element[] = []
  let report: IntersectionObserverCallback = () => {}
  class Observer {
    constructor(callback: IntersectionObserverCallback) {
      report = callback
    }
    observe = (el: Element) => void seen.push(el)
    unobserve = () => {}
    disconnect = () => {}
  }
  vi.stubGlobal('IntersectionObserver', Observer)
  return {
    show: () =>
      report(
        seen.map(
          (target) => ({ target, isIntersecting: true }) as never,
        ) as IntersectionObserverEntry[],
        {} as IntersectionObserver,
      ),
  }
}

beforeEach(() => fileText.mockClear())
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('a clip is a poster with a play badge and its duration', () => {
  const { container } = render(OutputTile, {
    props: {
      file: media({
        name: 'demo/clip.mp4',
        kind: 'video',
        label: 'clip.mp4',
        thumb: '/thumb/demo/clip.mp4',
        duration: 75,
      }),
    },
  })
  expect(container.querySelector('video')).toBeNull()
  const img = container.querySelector('img')!
  expect(img.getAttribute('src')).toBe('/thumb/demo/clip.mp4')
  expect(img.getAttribute('loading')).toBe('lazy')
  expect(container.querySelector('.videobadge')).not.toBeNull()
  expect(container.querySelector('.tile-time')?.textContent).toBe('1m 15s')
})

it('an image is letterboxed, never cropped', () => {
  const { container } = render(OutputTile, { props: { file: media({}) } })
  const img = container.querySelector('img')!
  expect(img.classList).toContain('tile-pic')
  // The rule the class carries, in patterns.css
  const patterns = readFileSync('src/patterns.css', 'utf8')
  const rule = patterns.match(/\.tile-pic\s*\{([^}]*)\}/)?.[1] ?? ''
  expect(rule).toMatch(/object-fit:\s*contain/)
  expect(rule).not.toMatch(/cover/)
})

it('audio shows its glyph, name and duration', () => {
  const { container } = render(OutputTile, {
    props: {
      file: media({
        name: 'demo/tone.wav',
        kind: 'audio',
        label: 'tone.wav',
        thumb: null,
        duration: 2.5,
      }),
      caption: 'none',
    },
  })
  expect(container.querySelector('img')).toBeNull()
  const face = container.querySelector('.tile-media')!
  expect(face.textContent).toContain('♪')
  expect(face.textContent).toContain('tone.wav')
  expect(face.textContent).toContain('2.5s')
})

it('a text tile reads its first lines once, when it is first seen', async () => {
  const observer = stubObserver()
  const { container } = render(OutputTile, {
    props: {
      file: media({
        name: 'demo/prompt.txt',
        kind: 'text',
        url: '/outputs/demo/prompt.txt',
        label: 'prompt.txt',
        thumb: null,
      }),
    },
  })
  await quiet()
  expect(fileText).not.toHaveBeenCalled()
  observer.show()
  await waitFor(() =>
    expect(container.querySelector('.tile-excerpt')?.textContent).toBe(
      'one\ntwo\nthree',
    ),
  )
  observer.show()
  await quiet()
  expect(fileText).toHaveBeenCalledTimes(1)
  expect(fileText).toHaveBeenCalledWith('/outputs/demo/prompt.txt')
  expect(container.querySelector('.tile-media')?.textContent).toContain('¶')
})

it('a text tile says (empty) for an empty file', async () => {
  const observer = stubObserver()
  fileText.mockResolvedValueOnce('')
  const { container } = render(OutputTile, {
    props: {
      file: media({ kind: 'text', label: 'blank.txt', thumb: null }),
    },
  })
  observer.show()
  await waitFor(() =>
    expect(container.querySelector('.tile-excerpt')?.textContent).toBe(
      '(empty)',
    ),
  )
})

it('a thumbnail that fails shows the kind glyph', async () => {
  const { container } = render(OutputTile, {
    props: {
      file: media({
        name: 'demo/clip.mp4',
        kind: 'video',
        label: 'clip.mp4',
        thumb: '/thumb/demo/clip.mp4',
      }),
    },
  })
  await fireEvent.error(container.querySelector('img')!)
  expect(container.querySelector('img')).toBeNull()
  expect(container.querySelector('.tile-glyph')).not.toBeNull()
  // Still a clip: the badge says it plays
  expect(container.querySelector('.videobadge')).not.toBeNull()
})

it('the star and note marks sit outside the caption', () => {
  const { container } = render(OutputTile, {
    props: {
      file: media({ starred: true, note: 'first line\nsecond' }),
      caption: 'clamp',
    },
  })
  const caption = container.querySelector('.tile-caption')!
  expect(caption.textContent?.trim()).toBe('a.png')
  const marks = container.querySelector('.tile-marks')!
  expect(caption.contains(marks)).toBe(false)
  expect(marks.querySelector('[aria-label="Starred"]')).not.toBeNull()
  const note = marks.querySelector('[aria-label="Has a note"]')!
  expect(note.getAttribute('title')).toBe('first line')
})

it('is a button with an onclick, a link with an href, and inert with neither', async () => {
  const onclick = vi.fn()
  const { container, unmount } = render(OutputTile, {
    props: { file: media({}), onclick, active: true, picked: true },
  })
  const button = container.querySelector('button.filetile')!
  expect(button.classList).toContain('active')
  expect(button.classList).toContain('picked')
  await fireEvent.click(button)
  expect(onclick).toHaveBeenCalledOnce()
  unmount()
  const link = render(OutputTile, { props: { file: media({}), href: '#/x' } })
  expect(link.container.querySelector('a.filetile')?.getAttribute('href')).toBe(
    '#/x',
  )
  link.unmount()
  const inert = render(OutputTile, { props: { file: media({}), dim: true } })
  expect(inert.container.querySelector('div.filetile.dim')).not.toBeNull()
})

it("a tile's accessible name is its file name, never its contents", async () => {
  const observer = stubObserver()
  const text = render(OutputTile, {
    props: {
      file: media({ kind: 'text', label: 'prompt.txt', thumb: null }),
      onclick: () => {},
    },
  })
  observer.show()
  await waitFor(() =>
    expect(text.container.querySelector('.tile-excerpt')).not.toBeNull(),
  )
  expect(
    text.container.querySelector('.tile-excerpt')?.getAttribute('aria-hidden'),
  ).toBe('true')
  expect(text.getByRole('button').textContent).toContain('one')
  text.getByRole('button', { name: 'prompt.txt' })
  text.unmount()
  const audio = render(OutputTile, {
    props: {
      file: media({ kind: 'audio', label: 'tone.wav', thumb: null }),
      onclick: () => {},
    },
  })
  // The face repeats the name the caption carries; only the caption is read
  audio.getByRole('button', { name: 'tone.wav' })
  audio.unmount()
  // With no caption, the face's name is the only one, and is read
  const bare = render(OutputTile, {
    props: {
      file: media({ kind: 'audio', label: 'tone.wav', thumb: null }),
      caption: 'none',
      onclick: () => {},
    },
  })
  bare.getByRole('button', { name: 'tone.wav' })
})

it("a text tile reads its lines again when the file's stamp changes", async () => {
  const observer = stubObserver()
  const text = (stamp: string) =>
    media({ kind: 'text', label: 'notes.txt', thumb: null, stamp })
  const { container, rerender } = render(OutputTile, {
    props: { file: text('1:10') },
  })
  observer.show()
  await waitFor(() =>
    expect(container.querySelector('.tile-excerpt')?.textContent).toBe(
      'one\ntwo\nthree',
    ),
  )
  // A listing read again hands the tile a new object for the same file
  await rerender({ file: text('1:10') })
  await quiet()
  expect(fileText).toHaveBeenCalledTimes(1)
  // Replaced: a new mtime and size
  fileText.mockResolvedValueOnce('new first\nnew second\n')
  await rerender({ file: text('2:22') })
  await waitFor(() =>
    expect(container.querySelector('.tile-excerpt')?.textContent).toBe(
      'new first\nnew second\n',
    ),
  )
  expect(fileText).toHaveBeenCalledTimes(2)
})

it('a row tile has no caption', () => {
  const { container } = render(OutputTile, {
    props: {
      file: media({ starred: true, note: 'keep' }),
      size: 'row',
      caption: 'none',
      alt: '',
    },
  })
  expect(container.querySelector('.tile-row')).not.toBeNull()
  expect(container.querySelector('.tile-caption')).toBeNull()
  expect(container.querySelector('.tile-marks')).toBeNull()
  const patterns = readFileSync('src/patterns.css', 'utf8')
  expect(patterns).toMatch(/\.tile-row\s*\{[^}]*width:\s*32px/)
})

it('a shadowed clip shows no play badge', () => {
  // ShadowedAssets' ghost: a name with no URL or poster, dimmed
  const { container } = render(OutputTile, {
    props: {
      file: media({ kind: 'video', url: '', thumb: null, label: 'clip.mp4' }),
      dim: true,
    },
  })
  expect(container.querySelector('.videobadge')).toBeNull()
  expect(container.querySelector('.tile-glyph')).not.toBeNull()
})

it('an asset tile carries its note mark', () => {
  const asset = assetMedia(
    {
      name: 'cast/priya.jpg',
      reference: 'asset:cast/priya.jpg',
      folder: 'cast',
      kind: 'image',
      size: 1,
      mtime: 1,
      origin: 'workspace',
      writable: true,
      url: '/inputs/cast/priya.jpg',
      note: 'the lead\nsecond',
    },
    'default',
  )
  const { container } = render(OutputTile, { props: { file: asset } })
  const note = container.querySelector('[aria-label="Has a note"]')!
  expect(note.getAttribute('title')).toBe('the lead')
})

it('an audio cover is 72 px, a picture cover 48', () => {
  const { container } = render(OutputTile, {
    props: {
      file: media({ kind: 'audio', thumb: null, label: 'tone.wav' }),
      size: 'mini',
      caption: 'none',
    },
  })
  expect(container.querySelector('.filetile')?.classList).toContain(
    'kind-audio',
  )
  const patterns = readFileSync('src/patterns.css', 'utf8')
  expect(patterns).toMatch(/\.tile-mini\s*\{\s*width:\s*48px;/)
  expect(patterns).toMatch(/\.tile-mini\.kind-audio\s*\{\s*width:\s*72px;/)
})

it('active, hover and picked look different', () => {
  // The rules for one selector alone, in patterns.css
  const patterns = readFileSync('src/patterns.css', 'utf8')
  const rule = (selector: string) => {
    const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    return patterns.match(
      new RegExp(`(?:^|\\n)${escaped}\\s*\\{([^}]*)\\}`),
    )?.[1]
  }
  // picked keeps its ink edge
  expect(rule('.filetile.picked')).toMatch(
    /box-shadow:\s*0 0 0 1px var\(--ink\)/,
  )
  // active is an accent outline, clear of the edge
  expect(rule('.filetile.active')).toMatch(
    /outline:\s*2px solid var\(--accent\)/,
  )
  // hover tints the panel and draws no edge or outline
  const hover = rule('a.filetile:hover,\nbutton.filetile:hover')
  expect(hover).toMatch(/background:/)
  expect(hover).not.toMatch(/border-color|outline|box-shadow/)
})

it('a tile that turns from an image into a text file reads its lines', async () => {
  const observer = stubObserver()
  const { container, rerender } = render(OutputTile, {
    props: { file: media({}) },
  })
  expect(fileText).not.toHaveBeenCalled()
  await rerender({
    file: media({
      name: 'demo/prompt.txt',
      kind: 'text',
      url: '/outputs/demo/prompt.txt',
      label: 'prompt.txt',
      thumb: null,
    }),
  })
  observer.show()
  await waitFor(() =>
    expect(container.querySelector('.tile-excerpt')?.textContent).toBe(
      'one\ntwo\nthree',
    ),
  )
})
