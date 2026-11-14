import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it } from 'vitest'
import TimelineStrip from './TimelineStrip.svelte'
import type { TimelineItem, TimelineSummary } from '../timeline'

afterEach(cleanup)

const item = (
  type: string,
  name: string,
  seconds: number,
  source?: string,
): TimelineItem => ({
  type,
  name,
  start_frame: 0,
  frames: seconds * 24,
  rate: 24,
  seconds,
  media: 'media/joined.mp4',
  ...(source !== undefined ? { source } : {}),
})

const summary = (items: TimelineItem[], audio = true): TimelineSummary => {
  const track = (kind: string) => ({
    kind,
    name: kind,
    clip_count: items.filter((i) => i.type === 'Clip').length,
    seconds: items.reduce((s, i) => s + (i.seconds ?? 0), 0),
    items,
  })
  return {
    kind: 'timeline',
    name: 'cut',
    otio_version: '0.17',
    has_audio_track: audio,
    // Audio listed first on purpose: the strip puts video on top
    tracks: audio ? [track('Audio'), track('Video')] : [track('Video')],
    members: ['content.otio', 'media/joined.mp4'],
  }
}

it('draws one block per clip with its label, video above audio', () => {
  const { container } = render(TimelineStrip, {
    summary: summary([
      item('Clip', 'opening', 2, 'output:w/20261108-101500-a1b2c3d4/j.mp4'),
      item('Gap', '', 1),
      item('Clip', 'closing', 3, 'output:w/20261108-101500-a1b2c3d4/j.mp4'),
    ]),
  })
  expect(screen.getAllByText('opening · 2 s')).toHaveLength(2)
  expect(screen.getAllByText('closing · 3 s')).toHaveLength(2)
  expect(container.querySelectorAll('.gap')).toHaveLength(2)
  const kinds = [...container.querySelectorAll('.track')].map(
    (t) => (t as HTMLElement).dataset.kind,
  )
  expect(kinds).toEqual(['Video', 'Audio'])
})

it('sizes each block by its share of the track', () => {
  const { container } = render(TimelineStrip, {
    summary: summary(
      [item('Clip', 'a', 1, 'output:x'), item('Clip', 'b', 3)],
      false,
    ),
  })
  const blocks = container.querySelectorAll<HTMLElement>('.block')
  expect(blocks[0].style.flexBasis).toBe('25%')
  expect(blocks[1].style.flexBasis).toBe('75%')
})

it('offers a copy button only for a clip that carries a source', () => {
  render(TimelineStrip, {
    summary: summary(
      [
        item('Clip', 'a', 1, 'output:w/20261108-101500-a1b2c3d4/j.mp4'),
        item('Clip', 'b', 1),
      ],
      false,
    ),
  })
  const buttons = screen.getAllByRole('button')
  expect(buttons).toHaveLength(1)
  expect(buttons[0].getAttribute('title')).toBe(
    'copy output:w/20261108-101500-a1b2c3d4/j.mp4',
  )
})

it('shows no copy button for a legacy summary without sources', () => {
  render(TimelineStrip, {
    summary: summary([item('Clip', 'a', 1), item('Clip', 'b', 1)]),
  })
  expect(screen.queryAllByRole('button')).toHaveLength(0)
})

it('lists as downloads the members that are the job files', () => {
  render(TimelineStrip, {
    summary: summary([item('Clip', 'a', 1)], false),
    memberHref: (m) => (m === 'media/joined.mp4' ? '/dl/joined.mp4' : null),
  })
  const link = screen.getByRole('link', { name: 'Download media/joined.mp4' })
  expect(link.getAttribute('href')).toBe('/dl/joined.mp4')
  expect(screen.queryByText('content.otio')).toBeNull()
})
