import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/svelte'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../test/apiMock'
import { feed } from '../test/feedMock'

const file = (name: string, kind = 'image') => ({
  name,
  label: name.split('/').pop(),
  kind,
  url: `/outputs/${name}`,
  folder: 'wf',
  subfolder: '',
  run_id: 'r',
  size: 1,
  mtime: 1,
})

const listing = vi.hoisted(() => ({ extra: [] as string[] }))
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
api.getJob.mockImplementation(() => Promise.reject(new Error('none')))

const defaults = () => {
  api.gallery.mockImplementation(() =>
    Promise.resolve({
      files: [
        file('wf/r1/a.png'),
        file('wf/r2/b.png'),
        file('wf/v1.mp4', 'video'),
        file('wf/v2.mp4', 'video'),
        ...listing.extra.map((n) =>
          file(n, n.endsWith('.wav') ? 'audio' : 'image'),
        ),
      ],
    }),
  )
  api.galleryMetadata.mockImplementation((name: string) =>
    Promise.resolve({
      metadata: {
        model_name: 'h3',
        seed: name.includes('a') ? 1 : 2,
        arguments: { steps: 12 },
      },
      job: null,
    }),
  )
}
defaults()

import ComparePage from './ComparePage.svelte'
import { compareHref } from '../routes'

describe('ComparePage', () => {
  afterEach(() => {
    cleanup()
    listing.extra = []
    api.gallery.mockReset()
    api.galleryMetadata.mockReset()
    defaults()
  })
  it('shows each file and only the argument that differs', async () => {
    render(ComparePage, { props: { names: ['wf/r1/a.png', 'wf/r2/b.png'] } })
    expect(await screen.findByAltText('a.png')).toBeTruthy()
    expect(await screen.findByRole('rowheader', { name: 'seed' })).toBeTruthy()
    expect(screen.queryByRole('rowheader', { name: 'steps' })).toBeNull()
  })
  it('with fewer than two files says how to pick them', async () => {
    render(ComparePage, { props: { names: ['wf/r1/a.png'] } })
    expect(await screen.findByText('Nothing to compare yet.')).toBeTruthy()
    expect(
      screen.getByText(
        'Tick two or more files in the Gallery, then press Compare.',
      ),
    ).toBeTruthy()
    expect(
      screen.getByRole('link', { name: 'Open the Gallery' }),
    ).toHaveAttribute('href', '#/ws/default/gallery')
    expect(screen.queryByAltText('a.png')).toBeNull()
  })
  it('says when a file is gone', async () => {
    render(ComparePage, { props: { names: ['wf/r1/a.png', 'wf/gone.png'] } })
    expect(await screen.findByText(/no longer in the gallery/)).toBeTruthy()
  })
  it('reads the clock from the first video, not the first cell', async () => {
    const { container } = render(ComparePage, {
      props: { names: ['wf/r1/a.png', 'wf/v1.mp4', 'wf/v2.mp4'] },
    })
    await screen.findByLabelText('position in every video')
    const [first, second] = container.querySelectorAll('video')
    const slider = screen.getByLabelText(
      'position in every video',
    ) as HTMLInputElement
    second.currentTime = 7
    await fireEvent(second, new Event('timeupdate'))
    expect(slider.value).toBe('0')
    first.currentTime = 3
    await fireEvent(first, new Event('timeupdate'))
    expect(screen.getByText('3.00 s')).toBeTruthy()
  })

  const names = ['wf/r1/a.png', 'wf/gone.png']
  const settle = () => new Promise((r) => setTimeout(r, 20))

  it('reloads when the gallery moves, with no skeleton, and cells keep their metadata', async () => {
    render(ComparePage, { props: { names } })
    await screen.findByAltText('a.png')
    await screen.findByText(/no longer in the gallery/)
    expect(feed.watched('default')).toEqual(['gallery'])
    const calls = api.galleryMetadata.mock.calls.length
    expect(calls).toBe(1)
    const reads = api.gallery.mock.calls.length

    // the missing file lands: only its cell asks for metadata
    listing.extra = ['wf/gone.png']
    feed.move('default', 'gallery')
    expect(screen.queryByLabelText('loading the comparison')).toBeNull()
    expect(screen.getByAltText('a.png')).toBeTruthy()
    await screen.findByAltText('gone.png')
    expect(api.gallery.mock.calls.length).toBe(reads + 1)
    expect(api.galleryMetadata).toHaveBeenCalledTimes(calls + 1)
    expect(api.galleryMetadata).toHaveBeenLastCalledWith('wf/gone.png')

    feed.move('default', 'gallery')
    await settle()
    expect(api.gallery.mock.calls.length).toBe(reads + 2)
    expect(api.galleryMetadata).toHaveBeenCalledTimes(calls + 1)
  })

  it('asks again for a cell whose metadata fetch failed', async () => {
    api.galleryMetadata.mockImplementationOnce(() =>
      Promise.reject(new Error('down')),
    )
    render(ComparePage, { props: { names: ['wf/r1/a.png', 'wf/r2/b.png'] } })
    await screen.findByAltText('a.png')
    expect(api.galleryMetadata).toHaveBeenCalledTimes(2)
    feed.move('default', 'gallery')
    await waitFor(() => expect(api.galleryMetadata).toHaveBeenCalledTimes(3))
  })

  it('keeps what is on screen when a quiet reload fails', async () => {
    render(ComparePage, { props: { names } })
    await screen.findByAltText('a.png')
    api.gallery.mockImplementationOnce(() => Promise.reject(new Error('down')))
    feed.move('default', 'gallery')
    await settle()
    expect(screen.getByAltText('a.png')).toBeTruthy()
  })

  it('does not reload for another kind or another workspace', async () => {
    render(ComparePage, { props: { names } })
    await screen.findByAltText('a.png')
    const reads = api.gallery.mock.calls.length
    feed.move('default', 'jobs')
    feed.move('elsewhere', 'gallery')
    await settle()
    expect(api.gallery.mock.calls.length).toBe(reads)
    await waitFor(() => expect(feed.watched('default')).toEqual(['gallery']))
  })

  it('with fewer than two names shows the empty copy without loading', async () => {
    render(ComparePage, { props: { names: ['wf/r1/a.png'] } })
    expect(screen.queryByLabelText('loading the comparison')).toBeNull()
    expect(await screen.findByText('Nothing to compare yet.')).toBeTruthy()
    await settle()
    expect(api.gallery).not.toHaveBeenCalled()
  })

  it('shows the load error when the gallery read fails with two names', async () => {
    api.gallery.mockImplementation(() => Promise.reject(new Error('down')))
    render(ComparePage, { props: { names: ['wf/r1/a.png', 'wf/r2/b.png'] } })
    expect(await screen.findByText(/Couldn't load the comparison/)).toBeTruthy()
  })

  it('a name holding a slash survives the round trip', async () => {
    const hash = compareHref('default', ['wf/r1/a.png', 'wf/r2/b.png'])
    const parts = hash.slice(2).split('/').map(decodeURIComponent)
    expect(parts.slice(3)).toEqual(['wf/r1/a.png', 'wf/r2/b.png'])
    render(ComparePage, { props: { names: parts.slice(3) } })
    expect(await screen.findByAltText('a.png')).toBeTruthy()
  })

  it('removing a cell rewrites the URL', async () => {
    location.hash = '#/ws/default/compare/x'
    render(ComparePage, {
      props: { names: ['wf/r1/a.png', 'wf/r2/b.png', 'wf/v1.mp4'] },
    })
    await screen.findByAltText('a.png')
    const [first] = screen.getAllByRole('button', {
      name: 'Remove from comparison',
    })
    await fireEvent.click(first)
    expect(location.hash).toBe(
      compareHref('default', ['wf/r2/b.png', 'wf/v1.mp4']),
    )
  })

  it('two same-leaf files from different workflows are told apart', async () => {
    listing.extra = ['wfA/r/a.png', 'wfB/r/a.png']
    render(ComparePage, { props: { names: ['wfA/r/a.png', 'wfB/r/a.png'] } })
    // the run is told by the version badge, so only the workflow is kept
    expect(
      await screen.findByRole('button', { name: 'Open wfA/a.png' }),
    ).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Open wfB/a.png' })).toBeTruthy()
    expect(
      screen.getByText('wfA/a.png', { selector: 'figcaption .mono' }),
    ).toBeTruthy()
  })

  it('a cell opens in the viewer', async () => {
    location.hash = '#/ws/default/compare/x'
    render(ComparePage, { props: { names: ['wf/r1/a.png', 'wf/r2/b.png'] } })
    await screen.findByAltText('a.png')
    await fireEvent.click(
      screen.getAllByRole('button', { name: 'Open a.png' })[0],
    )
    expect(location.hash).toContain('open=wf%2Fr1%2Fa.png')
    expect(await screen.findByRole('dialog', { name: 'a.png' })).toBeTruthy()
  })

  it('an open= naming no file compared is said, cleared and dismissed', async () => {
    location.hash = '#/ws/default/compare/x?open=wf%2Fr9%2Fgone.png'
    render(ComparePage, { props: { names: ['wf/r1/a.png', 'wf/r2/b.png'] } })
    const notice = await screen.findByText(
      'gone.png is not in this view any more.',
    )
    await waitFor(() => expect(location.hash).not.toContain('open='))
    expect(screen.queryByRole('dialog')).toBeNull()
    await fireEvent.click(
      within(notice).getByRole('button', { name: 'Dismiss' }),
    )
    expect(screen.queryByText(/is not in this view any more/)).toBeNull()
  })

  it('from a job, the back link returns to it', async () => {
    location.hash = compareHref('default', ['wf/r1/a.png', 'wf/r2/b.png'], {
      href: '#/ws/default/jobs/j1',
      label: 'Job',
    })
    render(ComparePage, { props: { names: ['wf/r1/a.png', 'wf/r2/b.png'] } })
    await screen.findByAltText('a.png')
    expect(screen.getByRole('link', { name: /Job/ })).toHaveAttribute(
      'href',
      '#/ws/default/jobs/j1',
    )
    // removing a cell keeps where it came from
    await fireEvent.click(
      screen.getAllByRole('button', { name: 'Remove from comparison' })[0],
    )
    expect(location.hash).toContain('back=')
  })
  it('with no back=, the back link is the Gallery', async () => {
    location.hash = '#/ws/default/compare/x'
    render(ComparePage, { props: { names: ['wf/r1/a.png', 'wf/r2/b.png'] } })
    await screen.findByAltText('a.png')
    expect(screen.getByRole('link', { name: /Gallery/ })).toHaveAttribute(
      'href',
      '#/ws/default/gallery',
    )
  })
  it('an audio cell opens in the viewer', async () => {
    listing.extra = ['wf/r1/s.wav']
    location.hash = '#/ws/default/compare/x'
    const { container } = render(ComparePage, {
      props: { names: ['wf/r1/a.png', 'wf/r1/s.wav'] },
    })
    await screen.findByAltText('a.png')
    expect(container.querySelector('audio')).not.toBeNull()
    await fireEvent.click(
      await screen.findByRole('button', { name: 'Open s.wav' }),
    )
    expect(location.hash).toContain('open=wf%2Fr1%2Fs.wav')
    expect(await screen.findByRole('dialog', { name: 's.wav' })).toBeTruthy()
  })
  it('the transport pauses under the viewer', async () => {
    location.hash = '#/ws/default/compare/x'
    const { container } = render(ComparePage, {
      props: { names: ['wf/v1.mp4', 'wf/v2.mp4'] },
    })
    await screen.findByLabelText('position in every video')
    const clips = [...container.querySelectorAll('video')]
    for (const clip of clips) {
      clip.play = vi.fn(() => Promise.resolve())
      clip.pause = vi.fn()
    }
    await fireEvent.click(screen.getByRole('button', { name: 'play all' }))
    await fireEvent.click(screen.getAllByRole('button', { name: /^Open v/ })[0])
    await waitFor(() => {
      for (const clip of clips) expect(clip.pause).toHaveBeenCalled()
    })
    expect(screen.getByRole('button', { name: 'play all' })).toBeTruthy()
  })
  it('one transport plays every clip, muted', async () => {
    const { container } = render(ComparePage, {
      props: { names: ['wf/v1.mp4', 'wf/v2.mp4'] },
    })
    await screen.findByLabelText('position in every video')
    const clips = [...container.querySelectorAll('video')]
    expect(clips).toHaveLength(2)
    for (const clip of clips) {
      expect(clip.muted).toBe(true)
      clip.play = vi.fn(() => Promise.resolve())
    }
    await fireEvent.click(screen.getByRole('button', { name: 'play all' }))
    for (const clip of clips) expect(clip.play).toHaveBeenCalled()
  })

  it('quiet reloads leave one listener per event on each clip', async () => {
    const add = vi.spyOn(HTMLVideoElement.prototype, 'addEventListener')
    const { container } = render(ComparePage, {
      props: { names: ['wf/v1.mp4', 'wf/v2.mp4'] },
    })
    await screen.findByLabelText('position in every video')
    const [first] = container.querySelectorAll('video')
    const mine = () =>
      add.mock.calls.filter(
        (c, n) =>
          add.mock.contexts[n] === first &&
          (c[0] === 'timeupdate' || c[0] === 'loadedmetadata'),
      ).length
    expect(mine()).toBe(2)
    for (let n = 0; n < 2; n++) {
      feed.move('default', 'gallery')
      await settle()
    }
    expect(container.querySelectorAll('video')[0]).toBe(first)
    expect(mine()).toBe(2)
    add.mockRestore()
  })

  // The router hands the page a new array of the same names on each
  // hashchange: the viewer opening (`?open=`) and Back closing it
  const playingPair = async () => {
    location.hash = '#/ws/default/compare/x'
    const props = () => ({ names: ['wf/v1.mp4', 'wf/v2.mp4'] })
    const view = render(ComparePage, { props: props() })
    await screen.findByLabelText('position in every video')
    const clips = [...view.container.querySelectorAll('video')]
    for (const clip of clips) {
      clip.play = vi.fn(() => Promise.resolve())
      clip.pause = vi.fn()
      Object.defineProperty(clip, 'duration', { value: 8 })
      await fireEvent(clip, new Event('loadedmetadata'))
    }
    await fireEvent.click(screen.getByRole('button', { name: 'play all' }))
    clips[0].currentTime = 3
    await fireEvent(clips[0], new Event('timeupdate'))
    const slider = screen.getByLabelText(
      'position in every video',
    ) as HTMLInputElement
    return { ...view, props, clips, slider }
  }

  it('the same names in a new array leave the transport and the load alone', async () => {
    const { rerender, props, clips, slider } = await playingPair()
    const reads = api.gallery.mock.calls.length
    await rerender(props())
    await settle()
    expect(screen.getByRole('button', { name: 'pause all' })).toBeTruthy()
    expect(slider.max).toBe('8')
    expect(slider.value).toBe('3')
    for (const clip of clips) expect(clip.pause).not.toHaveBeenCalled()
    expect(api.gallery.mock.calls.length).toBe(reads)
  })

  it('new names pause and rewind the clips with the transport', async () => {
    const { rerender, clips } = await playingPair()
    const reads = api.gallery.mock.calls.length
    await rerender({ names: ['wf/v1.mp4', 'wf/v2.mp4', 'wf/r1/a.png'] })
    await screen.findByAltText('a.png')
    expect(screen.getByRole('button', { name: 'play all' })).toBeTruthy()
    expect(screen.getByText('0.00 s')).toBeTruthy()
    for (const clip of clips) {
      expect(clip.pause).toHaveBeenCalled()
      expect(clip.currentTime).toBe(0)
    }
    expect(api.gallery.mock.calls.length).toBe(reads + 1)
  })
})
