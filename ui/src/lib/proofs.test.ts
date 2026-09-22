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
