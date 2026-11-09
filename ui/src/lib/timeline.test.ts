import { expect, it } from 'vitest'
import { isTimeline, label, shares, type TimelineTrack } from './timeline'

const item = (type: string, seconds: number | null, name = '') => ({
  type,
  name,
  start_frame: 0,
  frames: 0,
  rate: 24,
  seconds,
})

it('recognizes a timeline summary and nothing else', () => {
  expect(isTimeline({ kind: 'timeline', tracks: [] })).toBe(true)
  expect(isTimeline({ kind: 'video', tracks: [] })).toBe(false)
  expect(isTimeline({ kind: 'timeline' })).toBe(false)
  expect(isTimeline(null)).toBe(false)
  expect(isTimeline('timeline')).toBe(false)
})

it('shares sum to one with a gap counted', () => {
  const track: TimelineTrack = {
    kind: 'Video',
    name: 'Video',
    clip_count: 2,
    seconds: 4,
    items: [item('Clip', 1), item('Gap', 1), item('Clip', 2)],
  }
  const result = shares(track)
  expect(result).toEqual([0.25, 0.25, 0.5])
  expect(result.reduce((a, b) => a + b, 0)).toBe(1)
})

it('gives an empty track no width rather than dividing by zero', () => {
  const track: TimelineTrack = {
    kind: 'Video',
    name: 'Video',
    clip_count: 0,
    seconds: 0,
    items: [item('Gap', null)],
  }
  expect(shares(track)).toEqual([0])
})

it('labels an item with its name and seconds', () => {
  expect(label(item('Clip', 4.2, 'opening'))).toBe('opening · 4.2 s')
  expect(label(item('Clip', 2, 'b'))).toBe('b · 2 s')
  expect(label(item('Clip', null, 'c'))).toBe('c')
})
