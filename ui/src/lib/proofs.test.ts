import { expect, it } from 'vitest'
import { latestProofs } from './proofs'
import type { GalleryFile } from './types'

const file = (
  name: string,
  folder: string,
  kind: GalleryFile['kind'],
): GalleryFile => ({
  name,
  folder,
  subfolder: '',
  run_id: '',
  version: null,
  url: '/' + name,
  kind,
  size: 1,
  mtime: 1,
  label: name,
})

it('skips audio and text, so a workflow with neither image nor video has no proof', () => {
  const proofs = latestProofs([
    file('a.wav', 'song', 'audio'),
    file('a.txt', 'song', 'text'),
    file('n.txt', 'note', 'text'),
    file('n.mp4', 'note', 'video'),
  ])
  expect(proofs.song).toBeUndefined()
  expect(proofs.note.name).toBe('n.mp4')
})

it('keeps the first entry seen per folder, preferring an image over a video', () => {
  const proofs = latestProofs([
    file('a.mp4', 'shot', 'video'),
    file('a.png', 'shot', 'image'),
    file('b.png', 'still', 'image'),
    file('c.png', 'still', 'image'),
    file('loose.png', '', 'image'),
  ])
  expect(proofs.shot.name).toBe('a.png')
  expect(proofs.still.name).toBe('b.png')
  expect(Object.keys(proofs)).toEqual(['shot', 'still'])
})
