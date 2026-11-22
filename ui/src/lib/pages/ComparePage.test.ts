import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, describe, expect, it, vi } from 'vitest'
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
const gallery = vi.hoisted(() => vi.fn())
const galleryMetadata = vi.hoisted(() => vi.fn())
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))

vi.mock('../api', () => ({
  api: {
    gallery: () => gallery(),
    galleryMetadata: (name: string) => galleryMetadata(name),
    galleryThumbnailUrl: (n: string) => `/thumb/${n}`,
    getJob: () => Promise.reject(new Error('none')),
  },
}))

const defaults = () => {
  gallery.mockImplementation(() =>
    Promise.resolve({
      files: [
        file('wf/r1/a.png'),
        file('wf/r2/b.png'),
        file('wf/v1.mp4', 'video'),
        file('wf/v2.mp4', 'video'),
        ...listing.extra.map((n) => file(n)),
      ],
    }),
  )
  galleryMetadata.mockImplementation((name: string) =>
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

describe('ComparePage', () => {
  afterEach(() => {
    cleanup()
    listing.extra = []
    gallery.mockReset()
    galleryMetadata.mockReset()
    defaults()
  })
  it('shows each file and only the argument that differs', async () => {
    render(ComparePage, { props: { names: ['wf/r1/a.png', 'wf/r2/b.png'] } })
    expect(await screen.findByAltText('wf/r1/a.png')).toBeTruthy()
    expect(await screen.findByRole('rowheader', { name: 'seed' })).toBeTruthy()
    expect(screen.queryByRole('rowheader', { name: 'steps' })).toBeNull()
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
    await screen.findByAltText('wf/r1/a.png')
    await screen.findByText(/no longer in the gallery/)
    expect(feed.watched('default')).toEqual(['gallery'])
    const calls = galleryMetadata.mock.calls.length
    expect(calls).toBe(1)
    const reads = gallery.mock.calls.length

    // the missing file lands: only its cell asks for metadata
    listing.extra = ['wf/gone.png']
    feed.move('default', 'gallery')
    expect(screen.queryByLabelText('loading the comparison')).toBeNull()
    expect(screen.getByAltText('wf/r1/a.png')).toBeTruthy()
    await screen.findByAltText('wf/gone.png')
    expect(gallery.mock.calls.length).toBe(reads + 1)
    expect(galleryMetadata).toHaveBeenCalledTimes(calls + 1)
    expect(galleryMetadata).toHaveBeenLastCalledWith('wf/gone.png')

    feed.move('default', 'gallery')
    await settle()
    expect(gallery.mock.calls.length).toBe(reads + 2)
    expect(galleryMetadata).toHaveBeenCalledTimes(calls + 1)
  })

  it('keeps what is on screen when a quiet reload fails', async () => {
    render(ComparePage, { props: { names } })
    await screen.findByAltText('wf/r1/a.png')
    gallery.mockImplementationOnce(() => Promise.reject(new Error('down')))
    feed.move('default', 'gallery')
    await settle()
    expect(screen.getByAltText('wf/r1/a.png')).toBeTruthy()
  })

  it('does not reload for another kind or another workspace', async () => {
    render(ComparePage, { props: { names } })
    await screen.findByAltText('wf/r1/a.png')
    const reads = gallery.mock.calls.length
    feed.move('default', 'jobs')
    feed.move('elsewhere', 'gallery')
    await settle()
    expect(gallery.mock.calls.length).toBe(reads)
    await waitFor(() => expect(feed.watched('default')).toEqual(['gallery']))
  })
})
