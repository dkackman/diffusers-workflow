import { expect, it } from 'vitest'
import { latestProofs, proofByRun } from './proofs'
import type { GalleryFile } from './types'

const file = (
  name: string,
  folder: string,
  kind: GalleryFile['kind'],
  run_id = '',
): GalleryFile => ({
  name,
  folder,
  subfolder: '',
  run_id,
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

it("a run's proof is its first image, else its first clip", () => {
  const proofs = proofByRun([
    file('r1/a.wav', 'shot', 'audio', 'r1'),
    file('r1/a.mp4', 'shot', 'video', 'r1'),
    file('r1/b.mp4', 'shot', 'video', 'r1'),
    file('r1/a.png', 'shot', 'image', 'r1'),
    file('r1/b.png', 'shot', 'image', 'r1'),
    file('r2/c.mp4', 'shot', 'video', 'r2'),
    file('r2/d.mp4', 'shot', 'video', 'r2'),
    file('r3/n.txt', 'shot', 'text', 'r3'),
    file('loose.png', '', 'image'),
  ])
  expect(proofs.r1.name).toBe('r1/a.png')
  expect(proofs.r2.name).toBe('r2/c.mp4')
  expect(Object.keys(proofs)).toEqual(['r1', 'r2'])
})
